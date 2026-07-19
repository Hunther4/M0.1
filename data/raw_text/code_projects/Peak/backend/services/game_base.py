"""Shared game service base — eliminates duplication across memory, math, iq services.

Each game service (memory, math, iq) follows the same pattern for
iniciar_sesion and consolidar_sesion. This module extracts that common
logic so each service only provides game-specific callbacks.
"""
import json
from typing import Any, Callable, Optional

from sqlmodel import Session

from models.models import Skill, Session as PracticeSession


def iniciar_sesion_base(db: Session, SessionModel, skill_id: int):
    """Common session start: check skill exists, create game session, persist.

    Args:
        db: Database session
        SessionModel: The game-specific session model (MemoryNumberSession, etc.)
        skill_id: ID of the skill to practice

    Returns:
        The newly created game session instance
    """
    skill = db.get(Skill, skill_id)
    if not skill:
        raise ValueError("Skill not found")
    session = SessionModel(skill_id=skill_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def consolidar_sesion_base(
    db: Session,
    SessionModel,
    session_id: int,
    elapsed_seconds: Optional[int],
    game_name: str,
    build_session_data: Callable[[Any], dict],
    build_practice_fields: Callable[[Any], dict],
    min_rounds: int = 3,
    min_rounds_message: Optional[str] = None,
):
    """Common session consolidation: validate, close, create PracticeSession, link, apply staircase.

    Args:
        db: Database session
        SessionModel: The game-specific session model
        session_id: ID of the game session to consolidate
        elapsed_seconds: Optional elapsed time in seconds
        game_name: Name for error messages (e.g., "Math", "IQ")
        build_session_data: Callback that takes game_session → dict of session_data
        build_practice_fields: Callback that takes game_session → dict of PracticeSession fields
        min_rounds: Minimum rounds required (default 3)

    Returns:
        Tuple of (practice_session, game_session)
    """
    from services.staircase import (
        apply_staircase,
        compute_practice_level,
        determine_session_success,
    )

    game_session = db.get(SessionModel, session_id)
    if not game_session:
        raise ValueError(f"{game_name} session not found")
    if not game_session.is_active:
        raise ValueError("Session already consolidated")
    if game_session.total_rounds < min_rounds:
        msg = min_rounds_message or f"Minimum {min_rounds} rounds required before consolidation"
        raise ValueError(msg)

    game_session.is_active = False

    session_data = build_session_data(game_session)
    if elapsed_seconds is not None:
        session_data["elapsed_seconds"] = elapsed_seconds

    practice_fields = build_practice_fields(game_session)
    practice_fields["session_data"] = json.dumps(session_data)

    practice_session = PracticeSession(**practice_fields)
    db.add(practice_session)
    db.flush()

    game_session.consolidated_session_id = practice_session.id
    db.add(game_session)
    db.commit()
    db.refresh(practice_session)
    db.refresh(game_session)

    # Adaptive staircase (Ericsson 1↑/1↓)
    session_success = determine_session_success(db, game_session)
    skill = db.get(Skill, game_session.skill_id)
    if skill:
        apply_staircase(
            db,
            skill,
            session_success,
            trigger="session_consolidation",
            session_id=practice_session.id,
        )
        skill.practice_level = compute_practice_level(practice_session)
        db.add(skill)
        db.commit()

    return practice_session, game_session
