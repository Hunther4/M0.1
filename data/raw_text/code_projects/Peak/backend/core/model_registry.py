from typing import List

from sqlmodel import Session, select

from core.database import engine
from models.models import AiModel


def get_available_models(provider: str = None) -> List[AiModel]:
    """Retorna los modelos activos. Filtra por provider si se especifica."""
    with Session(engine) as session:
        query = select(AiModel).where(AiModel.is_active == True)  # noqa: E712
        if provider:
            query = query.where(AiModel.provider == provider)
        return list(session.exec(query).all())

def get_best_model_for_task(task_type: str, preferred_provider: str = None) -> AiModel:
    """
    Retorna el mejor modelo para la tarea (score más alto con la capability necesaria).
    1. Filtra por preferred_provider si está definido.
    2. Filtra por capabilities que incluyan task_type.
    3. Ordena por score DESC.
    """
    with Session(engine) as session:
        query = select(AiModel).where(AiModel.is_active == True)  # noqa: E712
        if preferred_provider:
            query = query.where(AiModel.provider == preferred_provider)

        models = session.exec(query).all()

        # Filtrado en memoria por capabilities (porque SQLite LIKE a veces es tricky)
        valid_models = [
            m for m in models
            if m.capabilities and task_type in m.capabilities.split("|")
        ]

        if not valid_models:
            # Fallback si no hay específicos de la tarea
            return get_default_model()

        valid_models.sort(key=lambda m: m.score, reverse=True)
        return valid_models[0]

def get_default_model() -> AiModel:
    """Retorna el modelo local LM Studio por defecto."""
    with Session(engine) as session:
        model = session.exec(select(AiModel).where(AiModel.provider == "lm_studio").where(AiModel.is_active == True)).first()  # noqa: E712
        if not model:
            # Fallback duro por si no corrieron el seed
            return AiModel(
                name="Local (Fallback)",
                provider="lm_studio",
                model_id="local-model",
                capabilities="audit|quick_log|assessment|general",
                score=50
            )
        return model
