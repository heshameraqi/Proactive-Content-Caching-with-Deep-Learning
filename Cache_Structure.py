from collections import OrderedDict
import pandas as pd
import random
import os


class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.cap_used = 0

    def get_cache(self):
        return self.cache.copy(), self.cap_used

    def cache_init(self, cache_l, cap_used_l):
        self.cache = cache_l.copy()
        self.cap_used = cap_used_l

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        self.cap_used += 1
        # lru_item = list(self.cache.keys())[0]
        while self.cap_used > self.capacity:
            self.cache.popitem(last=False)
            self.cap_used -= 1
            # lru_item = list(self.cache.keys())[0]

    def hit_ratio_lru(self, requests):
        hit, miss = 0, 0
        for item in requests:
            if self.get(item) != -1:
                hit += 1
            else:
                self.put(item, item)
                miss += 1
        hit_ratio_l = float(hit / (hit + miss))
        return hit_ratio_l


class MultiCache:
    def __init__(self, capacity, network_file, one_cache=False):
        if one_cache:  # in case we want to use only one cache for the simulation, in this case network_file will be ignored
            self.nb_nodes = 1
            self.parent_node_list = [-1]
            cache_lru = LRUCache(capacity)
            self.list_caches = list()
            self.user_node_list = list()
            self.list_caches.append(cache_lru)
            self.user_node_list.append(0)
        else:
            # each line the network_file will be in the following format: source_node_id -- destination_node_id
            df_network = pd.read_csv(os.path.join(os.getcwd(), 'data', network_file), sep='--', engine='python', encoding='latin-1',
                                     names=['source_node', 'destination_node'])
            # network nodes are identified starting from zero
            source_nodes = [int(i) for i in df_network['source_node'].tolist()]
            destination_nodes = [int(i) for i in df_network['destination_node'].tolist()]
            self.nb_nodes = len(list(dict.fromkeys(source_nodes + destination_nodes)))
            self.parent_node_list = [-1] * self.nb_nodes  # parent node index of each node (-1 if it is empty)
            for i in range(0, len(source_nodes)):
                self.parent_node_list[source_nodes[i]] = destination_nodes[i]
            self.list_caches = list()
            self.user_node_list = list()  # list of nodes (those without child nodes in the network) that will receive users requests
            for i in range(0, self.nb_nodes):
                cache_lru = LRUCache(capacity)
                self.list_caches.append(cache_lru)
                if i not in list(dict.fromkeys(destination_nodes)):
                    self.user_node_list.append(i)

    def get_cache(self):
        return self.list_caches.copy()

    def cache_init(self, list_cache_l):
        self.list_caches = list_cache_l

    def hit_ratio_multi_lru(self, requests):
        hit_g, miss_g = 0, 0  # to compute the overall cache hit (ratio of contents served by caches)
        hit_cache, miss_cache = ([0] * self.nb_nodes for _ in range(2))  # to compute the cache hit of each cache
        for item in requests:
            # print("current_request: ", item)
            # here we choose randomly which node will the receive the request, but it can be configured in a specific way
            current_node_index = random.choice(self.user_node_list)
            # current_node_index = 1
            found = False
            #  we start from the user node and go from one node to its parent until the content is found
            while not found and current_node_index != -1:
                current_cache = self.list_caches[current_node_index]
                # print("current_node_index: ", current_node_index, " / current_node_cache: ", current_cache.cache)
                if current_cache.get(item) != -1:
                    found = True
                    hit_g += 1
                    hit_cache[current_node_index] += 1
                    # print("hit_cache: ", hit_cache)
                else:
                    # here, we assume that every content will be cached in every node in case of a miss
                    current_cache.put(item, item)
                    miss_cache[current_node_index] += 1
                    parent_node_index = self.parent_node_list[current_node_index]
                    current_node_index = parent_node_index
                    # print("miss_cache: ", miss_cache)
            if not found:
                miss_g += 1
        return hit_g, miss_g, hit_cache, miss_cache

    def hit_ratio_multi_lru_cfl(self, requests, users, rec_movie_list):
        hit_g, miss_g = 0, 0  # to compute the overall cache hit (ratio of contents served by caches)
        hit_cache, miss_cache = ([0] * self.nb_nodes for _ in range(2))  # to compute the cache hit of each cache
        # for item in requests:
        for i in range(0, len(requests)):
            item = requests[i]
            # print("current_request: ", item)
            # here we choose randomly which node will the receive the request, but it can be configured in a specific way
            current_node_index = random.choice(self.user_node_list)
            # current_node_index = 1
            found = False
            #  we start from the user node and go from one node to its parent until the content is found
            while not found and current_node_index != -1:
                current_cache = self.list_caches[current_node_index]
                # print("current_node_index: ", current_node_index, " / current_node_cache: ", current_cache.cache)
                if current_cache.get(item) != -1:
                    found = True
                    hit_g += 1
                    hit_cache[current_node_index] += 1
                else:
                    if item in rec_movie_list:  # here we decide to cache the current request only if it exists in the recommendation list
                        current_cache.put(item, item)
                    miss_cache[current_node_index] += 1
                    parent_node_index = self.parent_node_list[current_node_index]
                    current_node_index = parent_node_index
            if not found:
                miss_g += 1
        return hit_g, miss_g, hit_cache, miss_cache


if __name__ == "__main__":
    print("hello")
    df_requests = pd.read_csv(os.path.join(os.getcwd(), 'data', 'ml-1m', 'ratings_new.dat'), sep='::', engine='python', encoding='latin-1',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # df_requests = pd.read_csv(os.path.join(os.getcwd(), 'data', 'requests_test.txt'), sep='::', engine='python', encoding='latin-1',
    #                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    requests = df_requests["movie_id"].tolist()
    users = df_requests["user_id"].tolist()
    network = MultiCache(2, 'network.txt', True)
    # hit_g, miss_g, hit_cache, miss_cache = network.hit_ratio_multi_lru_cfl(requests, users)
    hit_g, miss_g, hit_cache, miss_cache = network.hit_ratio_multi_lru(requests)
    hit_ratio_global = float(hit_g / (hit_g + miss_g))
    print("hit_ratio_global: ", hit_ratio_global)
