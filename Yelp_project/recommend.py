from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    
    centroid_candidate = centroids[0]
    for i in range(len(centroids) - 1):
        dist = distance(location, centroids[i + 1])
        if dist < distance(location, centroid_candidate):
            centroid_candidate = centroids[i + 1]
    return centroid_candidate

    


def group_by_first(pairs):
    """Return a list of lists that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)  # Values from pairs that start with 1, 3, and 2 respectively
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
   
    restaurants_all = []
    for i in restaurants:
        location = restaurant_location(i)
        dist = [distance(location, j) for j in centroids]
        closest_index = [dist.index(min(dist))]
        closest_index.append(i)
        restaurants_all.append(closest_index)

    return group_by_first(restaurants_all)

    


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    
    locations = [restaurant_location(i) for i in cluster]
    latitudes, longitudes = [i[0] for i in locations], [i[1] for i in locations]
    return [mean(latitudes), mean(longitudes)]
    


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0

    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        
        clusters_by_old_centroids = group_by_centroid(restaurants, old_centroids)
        centroids = [find_centroid(i) for i in clusters_by_old_centroids]
        
        n += 1
    return centroids



def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    xs = [feature_fn(r) for r in restaurants]
    ys = [user_rating(user, restaurant_name(r)) for r in restaurants]

    
    S_xx = sum([(i - mean(xs)) ** 2 for i in xs])
    S_yy = sum([(j - mean(ys)) ** 2 for j in ys])
    S_xy = sum([(i - mean(xs)) * (j - mean(ys)) for i, j in zip(xs, ys)])

    b = S_xy / S_xx
    a = mean(ys) - b * mean(xs)
    r_squared = (S_xy ** 2) / (S_xx * S_yy)
    

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    
    predictors_for_each_feature = [find_predictor(user, reviewed, i) for i in feature_fns]
    best_predictor = max(predictors_for_each_feature, key = lambda x: x[1])
    return best_predictor[0]
    


def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    
    def reviewed_or_not(r, reviewed):
        is_reviewed = [r == i for i in reviewed]
        if sum(is_reviewed):
            return True
        else:
            return False

    ratings_dict = {}
    for i in restaurants:
        r_name = restaurant_name(i)
        if reviewed_or_not(i, reviewed):
            ratings_dict[r_name] = user_rating(user, r_name)
        else:
            ratings_dict[r_name] = predictor(i)

    return ratings_dict


    


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    
    matched = []
    for i in restaurants:
        if sum(query == j for j in restaurant_categories(i)):
            matched.append(i)

    return matched
    


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)