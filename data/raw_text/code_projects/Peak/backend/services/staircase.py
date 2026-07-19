"""Adaptive staircase service — Ericsson's 1↑/1↓ algorithm.

Core functions:
  - apply_staircase: adjust skill level based on session success/failure
  - detect_plateau: detect when user is stuck (5+ sessions without level change)
  - compute_practice_level: derive Naive/Purposeful/Deliberate from session data
  - determine_session_success: extract success boolean from game session
"""

import logging
from sqlmodel import Session, select
from models.models import Skill, SkillLevelHistory

logger = logging.getLogger(__name__)

# Staircase window — how many recent results to consider
STAIRCASE_WINDOW = 3
PLATEAU_WINDOW = 5


def get_recent_results(db: Session, skill_id: int, window: int = STAIRCASE_WINDOW) -> list[bool]:
    """Get the success/failure results of the most recent sessions.

    Uses the explicit `success` field from SkillLevelHistory,
    NOT inferred from delta.
    """
    recent = db.exec(
        select(SkillLevelHistory)
        .where(SkillLevelHistory.skill_id == skill_id)
        .order_by(SkillLevelHistory.created_at.desc())
        .limit(window)
    ).all()

    # Reverse to chronological order (oldest first)
    recent.reverse()
    return [h.success for h in recent]


def apply_staircase(
    db: Session,
    skill: Skill,
    session_success: bool | None,
    trigger: str,
    session_id: int | None = None,
) -> None:
    """Apply the adaptive staircase algorithm (1↑/1↓).

    Rules:
      - 2+ recent successes (>=67%) → level up by 1
      - 2+ recent failures (<=33%) → level down by 1
      - Otherwise → maintain level

    Always records the result in SkillLevelHistory.

    Args:
        db: Database session
        skill: The skill to adjust
        session_success: True=success, False=failure, None=insufficient data
        trigger: What triggered this evaluation
        session_id: Optional linked session ID
    """
    if session_success is None:
        # No data — record but don't move level
        _record_history(db, skill, False, trigger, session_id)
        return

    recent = get_recent_results(db, skill.id, window=STAIRCASE_WINDOW)

    if len(recent) < 2:
        # First or second session — record but don't move level yet
        _record_history(db, skill, session_success, trigger, session_id)
        return

    # Include current result in calculation
    all_results = recent + [session_success]
    success_rate = sum(all_results) / len(all_results)
    old_level = skill.current_level

    if success_rate >= 0.67:  # 2/3 or 3/3 successful
        skill.current_level = min(skill.current_level + 1, 100)
    elif success_rate <= 0.33:  # 0/3 or 1/3 successful
        skill.current_level = max(skill.current_level - 1, 1)
    # else: maintain level

    _record_history(
        db, skill, session_success, trigger, session_id,
        old_level, skill.current_level,
    )


def _record_history(
    db: Session,
    skill: Skill,
    success: bool,
    trigger: str,
    session_id: int | None = None,
    old_level: float | None = None,
    new_level: float | None = None,
) -> SkillLevelHistory:
    """Record a staircase evaluation in SkillLevelHistory."""
    if old_level is None:
        old_level = skill.current_level
    if new_level is None:
        new_level = skill.current_level

    history = SkillLevelHistory(
        skill_id=skill.id,
        success=success,
        level_before=old_level,
        level_after=new_level,
        delta=new_level - old_level,
        trigger=trigger,
        session_id=session_id,
    )
    db.add(history)
    return history


def detect_plateau(db: Session, skill_id: int, window: int = PLATEAU_WINDOW) -> bool:
    """Detect if the user is in a plateau.

    Ericsson: plateaus are 1-2 weak sub-components, NOT absolute limits.
    We detect plateaus when the last `window` entries all have delta=0.
    """
    recent = db.exec(
        select(SkillLevelHistory)
        .where(SkillLevelHistory.skill_id == skill_id)
        .order_by(SkillLevelHistory.created_at.desc())
        .limit(window)
    ).all()

    return len(recent) >= window and all(h.delta == 0 for h in recent)


def compute_practice_level(session) -> str:
    """Derive the Three Levels of Practice from session data.

    Ericsson's framework:
      Level 1 — Naive: "I just practiced" (no structure)
      Level 2 — Purposeful: has goals, feedback, outside comfort zone
      Level 3 — Deliberate: all of the above + correction + planning

    Scoring:
      - what_i_practiced > 20 chars → +1 (has specific goal)
      - micro_error_found → +1 (self-monitoring)
      - correction_applied → +1 (feedback loop)
      - hypothesis_tomorrow → +1 (planning)
      - difficulty >= 4 → +1 (outside comfort zone)
      - score >= 4 → deliberate
      - score >= 2 → purposeful
      - else → naive
    """
    score = 0

    if session.what_i_practiced and len(session.what_i_practiced) > 20:
        score += 1

    if session.micro_error_found:
        score += 1

    if session.correction_applied:
        score += 1

    if session.hypothesis_tomorrow:
        score += 1

    if session.difficulty and session.difficulty >= 4:
        score += 1

    if score >= 4:
        return "deliberate"
    elif score >= 2:
        return "purposeful"
    return "naive"


def determine_session_success(db: Session, game_session, skill=None) -> bool | None:
    """Determine if a game session was successful.

    Returns True/False/None (None = insufficient data to decide).
    """
    from models.models import Skill

    if skill is None:
        # Fallback: look up from game_session.skill_id
        skill_id = getattr(game_session, 'skill_id', None)
        if skill_id is None:
            return None
        skill = db.get(Skill, skill_id)
    if not skill:
        return None

    # Games with consecutive_correct/consecutive_incorrect
    if hasattr(game_session, 'consecutive_correct'):
        cc = game_session.consecutive_correct
        ci = game_session.consecutive_incorrect

        if skill.skill_type == "memory_number":
            if cc >= 2:
                return True
            elif ci >= 2:
                return False
            return None

        elif skill.skill_type == "problem_set":
            if cc >= 3:
                return True
            elif ci >= 2:
                return False
            return None

        elif skill.skill_type == "iq_practice":
            if cc >= 2:
                return True
            elif ci >= 2:
                return False
            return None

    # DualNBack uses tasa_precision (NOT accuracy)
    if hasattr(game_session, 'tasa_precision'):
        if game_session.tasa_precision > 0.60:
            return True
        elif game_session.tasa_precision < 0.40:
            return False
        return None

    return None
