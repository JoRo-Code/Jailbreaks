def format_name(name: str, description: str) -> str:
    if description:
        return name+"-"+description
    else:
        return name