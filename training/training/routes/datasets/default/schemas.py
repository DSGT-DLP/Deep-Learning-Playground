from ninja import Schema


class DefaultDatasetResponse(Schema):
    data: list[str]
    message: str
