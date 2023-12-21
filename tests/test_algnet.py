import unittest
import jax.numpy as jnp
from algnet import batched_permutations, permute_along_bidders

class TestBatchedPermutations(unittest.TestCase):

    def test_batched_permutations(self):
        # Sample input array (adjust dimensions as needed)
        vals = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # add batch dimension such that shape is (1, num_bidders, num_items)
        vals = vals[None, ...]
        
        # Expected output for this input
        expected_output = jnp.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # no permutation for the first bidder
            [[4, 5, 6], [1, 2, 3], [7, 8, 9]],  # second bidder to front
            [[7, 8, 9], [1, 2, 3], [4, 5, 6]],  # third bidder to front
        ]).reshape(3, -1)

        # Flatten the expected output to compare with the flattened result
        num_bidders = 3 
        flattened_expected_output = expected_output.reshape(num_bidders, -1)

        result = batched_permutations(vals)

        # Check if the result matches the expected output
        self.assertTrue(jnp.array_equal(result, flattened_expected_output))
    
    def test_batched_vs_individual_permutations(self):
        # Sample input array (adjust dimensions as needed)
        vals = jnp.array([[1, 2, 3], [4, 5, 6]])

        # add batch dimension
        vals = vals[None, ...]

       # Using batched_permutations
        batched_result = batched_permutations(vals)

        # Using permute_along_bidders for each bidder
        num_bidders = vals.shape[1]
        individual_results = [permute_along_bidders(vals, i).ravel() for i in range(num_bidders)]
        individual_results_stacked = jnp.stack(individual_results, axis=0)

        # Check if the batched result matches the individual results
        self.assertTrue(jnp.array_equal(batched_result, individual_results_stacked))
    
    def test_single_bidder_case(self):
        # Sample input array (adjust dimensions as needed)
        vals = jnp.array([[1, 2, 3]])
        
        # add batch dimension
        vals = vals[None, ...]

        result = batched_permutations(vals)

        # Check if the result matches the expected output
        self.assertTrue(jnp.array_equal(result, vals[0, ...]))

if __name__ == '__main__':
    unittest.main()
