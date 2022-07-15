from enum import Enum, EnumMeta
from dataclasses import dataclass

class _BaseEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
    
class _BaseEnum(Enum, metaclass=_BaseEnumMeta):
    pass

def _process(cls, dataclass_class, kwargs):
    dataclass_fields = list(dataclass_class.__dataclass_fields__.keys())
    setattr(cls, 'Attribute', _BaseEnum('Attribute', [(field.upper(), field) for field in dataclass_fields]))
    
    for attribute in kwargs:
        if attribute in dataclass_fields:
            setattr(cls, attribute.capitalize(), _BaseEnum(attribute.capitalize(), [(element.upper(), element.upper()) for element in kwargs[attribute]]))
        else:
            del cls
            raise ValueError(f'{attribute} is not an attribute of the table')
    
    return cls
    

def enumclass(cls=None, /, *, dataclass_class=None, **kwargs):  
    def wrap(cls):
        return _process(cls, dataclass_class, kwargs)
    
    if cls is not None or dataclass_class is None:
        raise Exception('Please provide a corresponding dataclass')
    return wrap

# Usage
if __name__ == '__main__':
    @dataclass
    class StatusData:
        request_id: str
        status: str 
        timestamp: str

    @enumclass(dataclass_class=StatusData, status=['started', 'in_progress', 'success', 'failed'])
    class StatusEnums:
        pass
    
    print(StatusEnums)
    print(StatusEnums.Attribute)
    print(StatusEnums.Status)
    print(StatusEnums.Status.IN_PROGRESS.value)
    print('IN_PROGRESS' in StatusEnums.Status)