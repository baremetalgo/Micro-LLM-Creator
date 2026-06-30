def greet_user(name: str) -> str:
    if not name:
        return "Hello, developer"
    return f"Hello, {name}"


class Counter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> int:
        self.value += 1
        return self.value
