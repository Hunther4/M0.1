import json
import os
from functools import lru_cache
from typing import Optional

import yaml
from sqlmodel import Session

from core import router as core_router
from core.iq_practice import calculate_staircase, evaluate_attempt, generate_puzzle
from models.models import IQPracticeAttempt, IQPracticeRound, IQPracticeSession
from services.game_base import consolidar_sesion_base, iniciar_sesion_base
from services.learning_patterns_service import record_error


@lru_cache(maxsize=1)
def load_skill_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "skills", "iq-practice.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def iniciar_sesion(db: Session, skill_id: int) -> IQPracticeSession:
    return iniciar_sesion_base(db, IQPracticeSession, skill_id)


def crear_round(db: Session, session_id: int) -> tuple[IQPracticeRound, dict]:
    iq_session = db.get(IQPracticeSession, session_id)
    if not iq_session:
        raise ValueError("IQ session not found")
    if not iq_session.is_active:
        raise ValueError("IQ session is already closed")

    config = load_skill_config()
    puzzle = generate_puzzle(
        iq_session.level,
        config,
        core_router.execute_with_router,
        session_id=session_id,
        skill_id=iq_session.skill_id,
        db=db,
    )

    round_obj = IQPracticeRound(
        session_id=session_id,
        level=iq_session.level,
        puzzle_type=puzzle["puzzle_type"],
        question=puzzle["question"],
        options_json=json.dumps(puzzle["options"]),
        correct_answer=puzzle["correct_answer"],
        explanation=puzzle["explanation"],
    )
    db.add(round_obj)

    iq_session.total_rounds += 1
    db.add(iq_session)

    db.commit()
    db.refresh(round_obj)

    return round_obj, puzzle


def enviar_intento(db: Session, round_id: int, user_answer: str) -> dict:
    round_obj = db.get(IQPracticeRound, round_id)
    if not round_obj:
        raise ValueError("Round not found")

    iq_session = db.get(IQPracticeSession, round_obj.session_id)
    if not iq_session:
        raise ValueError("Session not found")

    correct = evaluate_attempt(user_answer, round_obj.correct_answer)

    staircase_result = calculate_staircase(
        iq_session.level,
        correct,
        iq_session.consecutive_correct,
        iq_session.consecutive_incorrect,
    )

    attempt = IQPracticeAttempt(
        round_id=round_id,
        user_answer=user_answer,
        correct=correct,
    )
    db.add(attempt)
    db.flush()  # get attempt.id for learning patterns

    # Record error pattern for adaptive AI (no-op if correct)
    if not correct:
        record_error(
            db,
            skill_id=iq_session.skill_id,
            skill_type="iq_practice",
            session_id=iq_session.id,
            round_id=round_id,
            attempt_id=attempt.id,
            is_correct=correct,
            user_answer=user_answer,
            correct_answer=round_obj.correct_answer,
            puzzle_type=round_obj.puzzle_type,
            level=iq_session.level,
        )

    iq_session.level = staircase_result["new_level"]
    iq_session.consecutive_correct = staircase_result["new_consecutive_correct"]
    iq_session.consecutive_incorrect = staircase_result["new_consecutive_incorrect"]

    if iq_session.level > iq_session.best_level:
        iq_session.best_level = iq_session.level

    db.add(iq_session)
    db.commit()
    db.refresh(attempt)

    return {
        "correct": correct,
        "correct_answer": round_obj.correct_answer if not correct else None,
        "explanation": round_obj.explanation,
        "staircase_result": staircase_result,
    }


def consolidar_sesion(db: Session, session_id: int, elapsed_seconds: Optional[int]) -> dict:
    def build_session_data(gs):
        data = {
            "type": "iq_practice",
            "total_rounds": gs.total_rounds,
            "best_level": gs.best_level,
            "final_level": gs.level,
        }
        if elapsed_seconds is not None:
            data["elapsed_seconds"] = elapsed_seconds
        return data

    def build_practice_fields(gs):
        return {
            "skill_id": gs.skill_id,
            "what_i_practiced": f"Práctica de IQ — Nivel {gs.level}",
            "micro_error_found": f"IQ session: {gs.total_rounds} puzzles, best level {gs.best_level}",
            "difficulty": 3,
            "entry_mode": "quick",
            "duration_minutes": max(10, gs.total_rounds * 2),
        }

    practice_session, gs = consolidar_sesion_base(
        db, IQPracticeSession, session_id, elapsed_seconds,
        "IQ", build_session_data, build_practice_fields,
    )

    return {
        "status": "consolidated",
        "practice_session_id": practice_session.id,
        "rounds_completed": gs.total_rounds,
        "best_level": gs.best_level,
        "summary": {
            "rounds_completed": gs.total_rounds,
            "best_level": gs.best_level,
        },
    }
