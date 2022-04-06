from typing import Callable, Sequence, Tuple
from enum import Enum
import itertools


class NumericalRelation(Enum):
  EQUAL = 0
  GREATER_EQUAL = 1
  # LESS = 2


class Equation:
  def __init__(self, num_var=0, a=None, b=None, op=None) -> None:
    '''
    Equation will have the following from
          a[0] * x_0 + a[1] * x_1 + b   op   0
          (if the number of variables is 2)

    num_var: number of variables
    a: a tuple of length num_var for specifying coefficients for each variable
    b: constant term
    op: relational operator ( >=, == ) specified by NumericalRelation
    '''
    self.num_var = num_var
    self.a = a
    self.b = b
    self.op = op


class RewardConstraints:
  def __init__(self) -> None:

    self.list_unary_constraints = []
    self.list_binary_constraints = []

    # elements are in the form of a tuple of (Equation, (input state, ...))
    self.manual_numerical_constraints = []

  def add_binary_constraint(self, condition: Callable, equation: Equation):
    '''
    condition: callback which takes two states as its input arguments
              and returns True or False
    equation: equation with a relational operater
    '''

    self.list_binary_constraints.append((condition, equation))

  def add_unary_constraint(self, condition: Callable, equation: Equation):
    '''
    condition: callback which takes a state as an input argument
              and returns True or False
    equation: equation with a relational operater
    '''

    self.list_unary_constraints.append((condition, equation))

  def add_numerical_constraints_manually(self, equation: Equation,
                                         input_states: Tuple):
    '''
    Add numerical constraints manually.
    This will help eliminate the effort to check conditions
      if we know individual states that we will impose constraints over.
    '''
    self.manual_numerical_constraints.append((equation, input_states))

  def get_numerical_constraints(self,
                                states) -> Sequence[Tuple[Equation, Tuple]]:
    '''
    states: states that possibly has constraints
    '''

    list_numerical_constraints = list(self.manual_numerical_constraints)

    if self.list_unary_constraints:
      for state in states:
        for cond, equ in self.list_unary_constraints:
          if cond(state):
            list_numerical_constraints.append((equ, (state, )))

    if self.list_binary_constraints:
      for state1, state2 in itertools.product(states, states):
        if state1 == state2:
          continue

        for cond, equ in self.list_binary_constraints:
          if cond(state1, state2):
            list_numerical_constraints.append((equ, (state1, state2)))

    return list_numerical_constraints
