def fitness(avg_speed, total_dist):
    return avg_speed + total_dist

def fitness_end(avg_speed, total_dist):
    return avg_speed * 10 + total_dist
