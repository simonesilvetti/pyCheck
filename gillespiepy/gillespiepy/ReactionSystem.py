import pandas as pd
import numpy as np
import importlib
from gillespiepy.gillespiepy.Reaction import Reaction
from gillespiepy.gillespiepy.Algorithms import algorithms_mapping
from typing import Dict, Tuple, List
import re
from functools import reduce
import pdb


class ReactionSystem:

    def __init__(self, parameters: pd.DataFrame, reactions: Dict[str, Reaction]):
        """

        :param parameters:
        :param reactions:
        """
        self.parameters = parameters
        self.reactions = reactions
        self.species_frame, self.constants_frame = self.split_parameters()
        self.species = np.array(self.species_frame['initial'])
        self.constants = np.array(self.constants_frame['initial'])
        self.input_matrix, self.output_matrix = self.create_stoichiometry_matrices()

        for reaction in self.reactions.values():
            reaction.rate_equation = self.rewrite_expressions(reaction.rate_equation)
        self.rate_expressions = ["rates[{}] = ".format(i) + r.rate_equation for i, r in enumerate(self.reactions.values())]
        self.number_of_reactions = len(self.reactions)
        self.number_of_species = len(self.species_frame)
        # self.reaction_affects = self.build_reaction_affects()
        # self.dependencies = self.build_dependencies()
        # self.dependency_graph = self.build_dependency_graph()
        # self.species_dependencies = self.build_species_dependencies()
        # self.reactant_indices, self.product_indices = self.build_reactants_and_products()

    def split_parameters(self):
        """
        Splits the parameters data frame into species and constants.
        :return: Tuple[pd.DataFrame, pd.DataFrame]      - The species and constants data frames.
        """
        reagent_filter = self.parameters['is_reagent']
        species_frame = self.parameters[reagent_filter][['name', 'initial', 'initialized', 'compartment']]
        constants_frame = self.parameters[~reagent_filter][['name', 'initial', 'initialized', 'compartment']]
        return species_frame, constants_frame

    def create_stoichiometry_matrices(self):
        """
        :return: Tuple[np.ndarray, np.ndarray]  - The input and output matrices
        """
        input_matrix = np.zeros((len(self.reactions), len(self.species_frame)))
        output_matrix = np.copy(input_matrix)

        for reaction_index, reaction in enumerate(self.reactions.values()):
            # Fill out input matrix
            for species, coefficient in reaction.inputs.items():
                species_index = self.species_frame.index.get_loc(species)
                input_matrix[reaction_index, species_index] = coefficient

            # Fill out output matrix
            for species, coefficient in reaction.outputs.items():
                species_index = self.species_frame.index.get_loc(species)
                output_matrix[reaction_index, species_index] = coefficient

        return input_matrix, output_matrix

    def __name_to_constant__(self, name: str):
        if name in self.constants_frame.index:
            new_name = str(self.constants_frame.loc[name]['initial'])
        elif name in self.species_frame.index:
            new_name = "species[{}]".format(self.species_frame.index.get_loc(name))
        else:
            new_name = name
        return new_name

    # TODO: Rewrite to use depth-first search, since Python doesn't like recursion.
    # TODO: Ideally rewrite this to use actual types.
    def rewrite_expressions(self, expression):
        """
        Rewrites all expression by replacing constants with their values, and species with array accesses.
        :param expression:
        :return:
        """
        if type(expression) == str:
            return self.__name_to_constant__(expression)
        # A function call. Replace the arguments list.
        elif len(expression) == 2:
            replaced_arguments = []
            for argument in expression[1]:
                replaced_arguments.append(self.rewrite_expressions(argument))
            return "{}({})".format(expression[0], ','.join(replaced_arguments))
        elif len(expression) == 3:
            if type(expression[1]) == list:
                return "{}{}{}".format('(', self.rewrite_expressions(expression[1]), ')')
            else:
                return "{}{}{}".format(self.rewrite_expressions(expression[0]),
                                       expression[1],
                                       self.rewrite_expressions(expression[2]))

    def full_expressions(self):
        full = []
        for index, rate_expression in enumerate(self.rate_expressions):
            full.append("rates[{}] = {}".format(index, rate_expression))
        return '\n'.join(full)

    def write_expression_to_file(self):
        strings = [
            "from numba import jit\n",
            "@jit(nopython=True)\n",
            "def calculate_rates(rates, species):\n"
        ]
        for index, expression in enumerate(self.rate_expressions):
            strings.append("  " + expression + "\n")
        strings.append("  return rates\n")
        with open("C:\/Users\Simone\Documents\phd\code\pycheck\gillespiepy\gillespiepy\/temp.py", "w+") as f:
            f.write(''.join(strings))
        func = importlib.import_module("gillespiepy.gillespiepy.temp")
        return func.calculate_rates

    #def run_simulation(self, end: int, method: str="direct", mode='steps'):
    def run_simulation(self,rates,runs, end: int, method: str="direct_parallel", mode='steps'):
        simulation_algorithm = algorithms_mapping[method]
        if mode == 'steps':
            # TODO: Set to max float
            end_time = 10**10
            end_steps = end
        else:
            end_time = end
            # TODO: Set to max int
            end_steps = 10**10
        self.write_expression_to_file()
       # expression = compile('\n'.join(self.rate_expressions), '<string>', 'exec')
        #func = self.write_expression_to_file()
        func = importlib.import_module("gillespiepy.gillespiepy.temp3")
        func=func.calculate_rates
        #return simulation_algorithm(end_time, end_steps, func, np.empty(len(self.rate_expressions)), self.species.copy(), self.output_matrix - self.input_matrix,20)
        #return simulation_algorithm(end_time, end_steps, func,np.tile(np.empty(len(self.rate_expressions)),(runs,1)), np.tile(self.species.copy(),(runs,1)), self.output_matrix - self.input_matrix,runs,rates)
        return simulation_algorithm(end_time, end_steps, func,np.tile(rates,(runs,1)), np.tile(self.species.copy(),(runs,1)), self.output_matrix - self.input_matrix,runs)
