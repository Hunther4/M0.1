import json
from typing import List, Optional

from pydantic import BaseModel
from sqlmodel import Session, select

from core import router as core_router
from core.memory_number import (
    calculate_staircase,
    evaluate_attempt,
    generate_numbers,
    get_phase_config,
)
from models.cognitive_models import MemorySessionMeta, MemoryStrategyLog
from models.models import MemoryNumberAttempt, MemoryNumberRound, MemoryNumberSession
from services.game_base import consolidar_sesion_base, iniciar_sesion_base
from services.learning_patterns_service import record_error


class CoachingResponse(BaseModel):
    coaching_message: str


def iniciar_sesion(db: Session, skill_id: int) -> MemoryNumberSession:
    return iniciar_sesion_base(db, MemoryNumberSession, skill_id)


def crear_round(db: Session, session_id: int) -> tuple[MemoryNumberRound, tuple]:
    game_session = db.get(MemoryNumberSession, session_id)
    if not game_session:
        raise ValueError("Game session not found")
    if not game_session.is_active:
        raise ValueError("Game session is already closed")

    phase_config = get_phase_config(game_session.phase)
    numbers = generate_numbers(
        game_session.current_span,
        phase_config["digit_max"],
        phase_config.get("ai_assisted", False),
    )

    round_obj = MemoryNumberRound(
        game_session_id=session_id,
        phase=game_session.phase,
        span=game_session.current_span,
        digit_max=phase_config["digit_max"],
        numbers_json=json.dumps(numbers),
        sequence_length=len(numbers),
        ai_assisted=phase_config.get("ai_assisted", False),
    )
    db.add(round_obj)

    game_session.total_rounds += 1
    db.add(game_session)

    db.commit()
    db.refresh(round_obj)

    return round_obj, (numbers, phase_config)


def enviar_intento(db: Session, round_id: int, submitted_numbers: List[int]) -> dict:
    round_obj = db.get(MemoryNumberRound, round_id)
    if not round_obj:
        raise ValueError("Round not found")

    expected = json.loads(round_obj.numbers_json)
    evaluation = evaluate_attempt(expected, submitted_numbers)

    attempt = MemoryNumberAttempt(
        round_id=round_id,
        submitted_numbers_json=json.dumps(submitted_numbers),
        correct=evaluation["correct"],
        correct_positions=evaluation["correct_positions"],
        total_positions=evaluation["total_positions"],
        errors_json=json.dumps(evaluation["errors"]) if evaluation["errors"] else None,
    )
    db.add(attempt)
    db.flush()  # get attempt.id for learning patterns

    # Record error pattern for adaptive AI (no-op if correct)
    game_session = db.get(MemoryNumberSession, round_obj.game_session_id)
    if not evaluation["correct"] and game_session:
        record_error(
            db,
            skill_id=game_session.skill_id,
            skill_type="memory_number",
            session_id=game_session.id,
            round_id=round_id,
            attempt_id=attempt.id,
            is_correct=False,
            user_answer=submitted_numbers,
            correct_answer=expected,
            level=game_session.current_span,
        )

    # Default result so the response stays well-formed even if session is missing
    result = {
        "new_span": round_obj.span,
        "new_phase": round_obj.phase,
        "phase_changed": False,
        "message": "",
        "new_consecutive_correct": 0,
        "new_consecutive_incorrect": 0,
    }

    if game_session:
        result = calculate_staircase(
            game_session.current_span,
            game_session.phase,
            evaluation["correct"],
            game_session.consecutive_correct,
            game_session.consecutive_incorrect,
        )
        game_session.current_span = result["new_span"]
        game_session.consecutive_correct = result["new_consecutive_correct"]
        game_session.consecutive_incorrect = result["new_consecutive_incorrect"]

        if result["phase_changed"]:
            game_session.phase = result["new_phase"]

        if game_session.current_span > game_session.best_span:
            game_session.best_span = game_session.current_span
        if game_session.phase > game_session.best_phase:
            game_session.best_phase = game_session.phase

        db.add(game_session)

    db.commit()
    db.refresh(attempt)

    return {
        "id": attempt.id,
        "correct": evaluation["correct"],
        "correct_positions": evaluation["correct_positions"],
        "total_positions": evaluation["total_positions"],
        "errors": evaluation["errors"],
        "staircase_result": result,
        "next_timing": get_phase_config(result.get("new_phase", round_obj.phase))["timing"],
    }


def consolidar_sesion(db: Session, session_id: int, elapsed_seconds: Optional[int]) -> dict:
    # Memory-specific analysis (read-only, before consolidation)
    game_session = db.get(MemoryNumberSession, session_id)
    if not game_session:
        raise ValueError("Game session not found")

    rounds = db.exec(
        select(MemoryNumberRound)
        .where(MemoryNumberRound.game_session_id == session_id)
        .order_by(MemoryNumberRound.created_at)
    ).all()

    round_ids = [r.id for r in rounds]
    attempts_by_round = {}
    if round_ids:
        all_attempts = db.exec(
            select(MemoryNumberAttempt)
            .where(MemoryNumberAttempt.round_id.in_(round_ids))
        ).all()
        for a in all_attempts:
            attempts_by_round[a.round_id] = a

    # Analyze performance per span
    span_stats = {}
    for r in rounds:
        span = r.span
        if span not in span_stats:
            span_stats[span] = {"total": 0, "correct": 0}
        span_stats[span]["total"] += 1
        attempt = attempts_by_round.get(r.id)
        if attempt and attempt.correct:
            span_stats[span]["correct"] += 1

    weak_spans = []
    strong_spans = []
    for span, stats in sorted(span_stats.items()):
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        if accuracy < 0.6:
            weak_spans.append(f"Span {span}: {stats['correct']}/{stats['total']} correctos")
        elif accuracy >= 0.8:
            strong_spans.append(f"Span {span}: {stats['correct']}/{stats['total']} correctos")

    coaching_message = _generate_coaching(
        weak_spans, strong_spans, game_session.best_span, game_session.best_phase
    )

    def build_session_data(gs):
        data = {
            "type": "memory_number",
            "total_rounds": gs.total_rounds,
            "best_span": gs.best_span,
            "best_phase": gs.best_phase,
            "final_phase": gs.phase,
            "final_span": gs.current_span,
        }
        if elapsed_seconds is not None:
            data["elapsed_seconds"] = elapsed_seconds
        return data

    def build_practice_fields(gs):
        return {
            "skill_id": gs.skill_id,
            "what_i_practiced": f"Memorizar Números — Fase {gs.phase}, Span {gs.current_span}",
            "micro_error_found": f"Game session: {gs.total_rounds} rounds, best span {gs.best_span}",
            "difficulty": 3,
            "entry_mode": "quick",
            "duration_minutes": max(10, gs.total_rounds * 2),
        }

    practice_session, gs = consolidar_sesion_base(
        db, MemoryNumberSession, session_id, elapsed_seconds,
        "Memory", build_session_data, build_practice_fields,
    )

    return {
        "status": "consolidated",
        "practice_session_id": practice_session.id,
        "rounds_completed": gs.total_rounds,
        "best_span": gs.best_span,
        "best_phase": gs.best_phase,
        "coaching_message": coaching_message,
        "summary": {
            "rounds_completed": gs.total_rounds,
            "best_span": gs.best_span,
            "best_phase": gs.best_phase,
            "coaching_message": coaching_message,
        },
    }


def _generate_coaching(weak_spans: list, strong_spans: list, best_span: int, best_phase: int) -> str:
    """Generate a brief AI coaching message based on session performance."""
    if not weak_spans and not strong_spans:
        return "Seguí practicando para mejorar tu memoria de trabajo."

    performance_summary = ""
    if weak_spans:
        performance_summary += f"Niveles con dificultad: {', '.join(weak_spans)}. "
    if strong_spans:
        performance_summary += f"Niveles dominados: {', '.join(strong_spans)}. "

    system_prompt = (
        "Coach memoria de trabajo. Español rioplatense, cálido, directo. "
        "Máx 2-3 oraciones. Estrategia CONCRETA + ACCIONABLE.\n\n"
        "✅ BUENO: 'Agrupá de a 3 números: recordás 3 bloques, no 9 sueltos. Probá en la próxima ronda.'\n"
        "✅ BUENO: 'Cuando falles span 6, bajá a 5 y hacé 3 rondas perfectas antes de subir.'\n"
        "❌ MALO: 'Tu memoria de trabajo es clave para aprender.'\n"
        "❌ MALO: 'Seguí practicando con constancia.'\n\n"
        "REGLAS:\n"
        "- NO expliques ciencia. NO des vueltas.\n"
        "- SÍ: qué HACER distinto mañana.\n"
        "- Si no hay weak_spans: felicitá + proponé siguiente reto.\n"
        "- Respondé SOLO el texto, sin formato adicional."
    )
    user_prompt = (
        f"Mejor span: {best_span}. Fase: {best_phase}.\n"
        f"{performance_summary}\n"
        "Consejo accionable:"
    )

    try:
        result = core_router.execute_with_router(
            task_type="memory_coaching",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=CoachingResponse,
        )
        return result.coaching_message if result else _coaching_fallback(weak_spans)
    except Exception:
        return _coaching_fallback(weak_spans)


def _coaching_fallback(weak_spans: list) -> str:
    """Fallback coaching when AI is unavailable."""
    if weak_spans:
        return "Noté que te costaron los niveles más altos. Seguí practicando con constancia — la memoria de trabajo se mejora con repetición."
    return "¡Buen trabajo! Tu memoria de trabajo está funcionando bien. Seguí desafiándote con spans más altos."


def log_round_strategy(db: Session, round_id: int, strategy_used: str) -> dict:
    """Insert a MemoryStrategyLog row for a completed round attempt."""
    round_obj = db.get(MemoryNumberRound, round_id)
    if not round_obj:
        raise ValueError("Round not found")

    attempts = db.exec(
        select(MemoryNumberAttempt).where(MemoryNumberAttempt.round_id == round_id)
    ).all()
    effective = any(a.correct and a.correct_positions == a.total_positions for a in attempts)

    log = MemoryStrategyLog(
        game_session_id=round_obj.game_session_id,
        round_id=round_id,
        strategy_used=strategy_used,
        effective=effective,
        span_at_attempt=round_obj.span,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return {"id": log.id, "effective": effective}


def save_session_meta(
    db: Session,
    session_id: int,
    strategy_type: str,
    self_reported_difficulty: Optional[int],
    notes: Optional[str],
) -> dict:
    """Insert a MemorySessionMeta row for a finalized game session."""
    game_session = db.get(MemoryNumberSession, session_id)
    if not game_session:
        raise ValueError("Game session not found")
    if self_reported_difficulty is not None and not (1 <= self_reported_difficulty <= 5):
        raise ValueError("difficulty must be 1-5")

    meta = MemorySessionMeta(
        game_session_id=session_id,
        strategy_type=strategy_type,
        self_reported_difficulty=self_reported_difficulty,
        notes=notes,
    )
    db.add(meta)
    db.commit()
    db.refresh(meta)
    return {
        "id": meta.id,
        "strategy_type": meta.strategy_type,
        "self_reported_difficulty": meta.self_reported_difficulty,
    }
