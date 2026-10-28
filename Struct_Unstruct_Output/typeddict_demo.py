from typing import TypedDict
class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name':'Gyan','age':23}   
new_person1: Person = {'name':'Gyan','age':'23'}
print(new_person)
print(new_person1)
