from ninja import Schema


class NotFoundError(Schema):
    message: str
