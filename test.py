class Parent:
    def show(self):
        print("This is the Parent class method.")

class Child(Parent):
    def show(self):
        # Call the parent class's show method first
        super().show()
        # Now add additional functionality
        print("This is the Child class method extending the Parent class method.")

# Usage
parent = Parent()
child = Child()

#parent.show()  # Calls the parent class method
child.show()   # Calls the overridden method in the child class
