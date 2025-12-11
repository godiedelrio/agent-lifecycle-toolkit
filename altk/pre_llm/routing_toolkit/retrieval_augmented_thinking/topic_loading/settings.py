from pydantic import BaseModel
from altk.pre_llm.routing_toolkit.retrieval_augmented_thinking.topic_extractor.settings import (
    TopicSinkSettings,
)


class TopicLoadingSettings(BaseModel):
    topics_sink: TopicSinkSettings | None = None
