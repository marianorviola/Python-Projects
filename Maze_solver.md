# Maze Solver
>In this program, we build a Maze solver API that uses object oriented programming (OOP) to construct a maze class and a maze object.

**Description of the Maze class:**

- The maze is a class object with both a start point and an end point. 
- It utilizes a stack data structure to store the elements of the path that it traverses.
- The maze class also applies backtracking and iteration.
- We chose to use iteration instead of recursion because
iteration is more readable and more intuitive to understand
than recursion. 
- The maze also uses a python "decorator" for the maze class solution.

**Description of the Maze object:**

- The user specifies the size of the maze (i.e., the number of rows and columns). 
- The maze object has a valid path (indicated by a "1") which can be traversed. 
- An invalid index or path (indicated by a "0") is like a wall that blocks the path. 
- The user specifies the path that is either valid or invalid by assigning a "1" or a "0" to the cell. 
- The maze structure has 4 directions that the program can traverse in. It is followed in the order of: 1) left, 2) up, 3) right and 4) down. 

**Final Outcome:**

- If the final outcome is a SUCCESS, the traversal path from start point to end point and the total number of moves that it took to reach the end-point, is printed. 
- If the program FAILS to escape the maze because the user-specified end point is a non-valid index or path, then it starts to back-track. 
- Every time it back-tracks, it prints the current path that it takes and pops it off the stack with the message: "popping off the stack". 
- Once it reaches the initial start point, it prints a "0" integer indicating that the user-specified end-point is a non-valid path or index. 
- The program then terminates. 

### Step 1: Create the maze template with attributes.

We import the "named tuple" module from python's "collections" library. The "named tuple" method identifies the current position (row, col) of our location within the maze object.

```python
from collections import namedtuple

class Maze:
    def __init__(self, maze = None, start = None, end = None):
        self.maze = maze 
        self.start = start
        self.end = end
```
### Step 2: Print the maze structure. 

Because the maze uses a string format, we will apply method override procedures to insert elements that use a "non-string" format. 

```python
    def __str__(self):
        data = ''
        for i, row in enumerate(self.maze):
            if i == 0:
                data += '-' * (len(row) +2) + '\n'
            for j, col in enumerate(row):
                if j == 0:
                    data += '|'
                if (i, j) == (self.start.row, self.start.col):
                    data += '*'
                elif (i, j) == (self.end.row, self.end.col):
                    data += '*'
                elif col == 1:
                    data += ' '
                else:
                    data += 'X'
                if j == len(row) - 1:
                    data += '|'
            data += '\n'
            if i == len(row) - 1:
                data += '-' * (len(row) +2) + '\n'
        return data
```
### Step 3: Define the conditions for a valid path.

An index with the element "1" indicates a valid path while an index with the element "0" indicates a "wall" or a non-valid path. 

```python
    def valid_index(self, position):
        if position.col < 0 or position.col >= len(self.maze[0]):
            return False
        elif position.row < 0 or position.row >= len(self.maze):
            return False
        elif self.maze[position.row][position.col] == 0:
            return False
        return True
```

### Step 4: Define the path directions that the program can traverse on within the maze. 

- The program can traverse in 4 directions within the following order: 1) left, 2) up, 3) right and 4) down. 
- We create a decorator in our program using the "@property" notation. 

```python
    @property 
    def solution(self):
        current = self.start
        path = [current]
  
        while True:
            current = path[-1]
            self.maze[current.row][current.col] = 0
            print(current, self.end)
            #print(self)
            if current == self.end:
                return path

            # move left
            elif self.valid_index(Maze.Position(row = current.row, col = current.col - 1)):
                current = Maze.Position(row = current.row, col = current.col - 1)
                path.append(current)
                print('moving left')
                continue
            # up
            elif self.valid_index(Maze.Position(row = current.row - 1, col = current.col)):
                current = Maze.Position(row = current.row - 1, col = current.col)
                path.append(current)
                print('moving up')
                continue
            # right
            elif self.valid_index(Maze.Position(row = current.row, col = current.col + 1)):
                current = Maze.Position(row = current.row, col = current.col + 1)
                path.append(current)
                print('moving right')
                continue
            # down
            elif self.valid_index(Maze.Position(row = current.row + 1, col = current.col)):
                current = Maze.Position(row = current.row + 1, col = current.col)
                path.append(current)
                print('moving down')
                continue

            path.pop()
            print('Popping off stack')
            if len(path) == 0:
                return []

    Position = namedtuple('Position', ['row', 'col'])

start = Maze.Position(row=0, col=1)
end = Maze.Position(row = 0, col = 0)
```

### Step 5: Create the user-defined maze structure.

- The user creates the maze structure by defining how many rows and columns he/she wants. 
- Moreover, the user also specifies which cell or index is valid and which is not valid (1 = path, 0 = wall). 

```python
maze_structure = [
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

maze = Maze(maze_structure, start, end)

print(len(maze.solution))
```

### **Scenario 1: Success! - User escapes the maze.**

- Program reaches the specified end-point successfully and escapes the maze.
- In "start" variable, specify a starting position of row = 0, col = 1
- In "end" variable, specify an end position of row = 3, col = 3

```python
start = Maze.Position(row=0, col=1)
end = Maze.Position(row = 3, col = 3)
```

**Final Output is a success!**

- Maze Solver shows the current path with 12 iterations from start point to end point. 

>Position(row=0, col=1) Position(row=3, col=3)  
moving right  
Position(row=0, col=2) Position(row=3, col=3)  
moving right  
Position(row=0, col=3) Position(row=3, col=3)  
moving down  
Position(row=1, col=3) Position(row=3, col=3)  
moving left  
Position(row=1, col=2) Position(row=3, col=3)  
moving down  
Position(row=2, col=2) Position(row=3, col=3)  
moving left  
Position(row=2, col=1) Position(row=3, col=3)  
moving left  
Position(row=2, col=0) Position(row=3, col=3)  
moving down  
Position(row=3, col=0) Position(row=3, col=3)  
moving right  
Position(row=3, col=1) Position(row=3, col=3)  
moving right  
Position(row=3, col=2) Position(row=3, col=3)  
moving right  
Position(row=3, col=3) Position(row=3, col=3)  
12

### **Scenario 2: Fail! - User remains trapped in the maze.**

- Program reaches an invalid path at the end-point - the user remains TRAPPED in the maze.
- In "start" variable, specify a starting position of row = 0, col = 1
- In "end" variable, specify an end position of row = 0, col = 0
- As observed in the maze structure, the end-point is not a valid index so the program prints a "0" integer.

```python
start = Maze.Position(row=0, col=1)
end = Maze.Position(row = 0, col = 0)
```

**Final Output is a Fail!** 

- Maze Solver shows the current path that leads to a non-valid index. 
- It back-tracks and pops off the stack each position that it has taken.
- Once it reaches the initial start point, it prints a "0" integer indicating a non-valid path.

>Position(row=0, col=1) Position(row=0, col=0)  
moving right  
Position(row=0, col=2) Position(row=0, col=0)  
moving right  
Position(row=0, col=3) Position(row=0, col=0)  
moving down  
Position(row=1, col=3) Position(row=0, col=0)  
moving left  
Position(row=1, col=2) Position(row=0, col=0)  
moving down  
Position(row=2, col=2) Position(row=0, col=0)  
moving left  
Position(row=2, col=1) Position(row=0, col=0)  
moving left  
Position(row=2, col=0) Position(row=0, col=0)  
moving down  
Position(row=3, col=0) Position(row=0, col=0)  
moving right  
Position(row=3, col=1) Position(row=0, col=0)  
moving right  
Position(row=3, col=2) Position(row=0, col=0)  
moving right  
Position(row=3, col=3) Position(row=0, col=0)  
Popping off stack  
Position(row=3, col=2) Position(row=0, col=0)  
Popping off stack  
Position(row=3, col=1) Position(row=0, col=0)  
Popping off stack  
Position(row=3, col=0) Position(row=0, col=0)  
Popping off stack  
Position(row=2, col=0) Position(row=0, col=0)  
Popping off stack  
Position(row=2, col=1) Position(row=0, col=0)  
Popping off stack  
Position(row=2, col=2) Position(row=0, col=0)  
Popping off stack  
Position(row=1, col=2) Position(row=0, col=0)  
Popping off stack  
Position(row=1, col=3) Position(row=0, col=0)  
Popping off stack  
Position(row=0, col=3) Position(row=0, col=0)  
Popping off stack  
Position(row=0, col=2) Position(row=0, col=0)  
Popping off stack  
Position(row=0, col=1) Position(row=0, col=0)  
Popping off stack  
0
