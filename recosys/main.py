from recosys.recommender import RandomRecommender
import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    RandomRecommender().run_sample()
