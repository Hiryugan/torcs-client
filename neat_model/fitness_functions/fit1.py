def fitness(avg_speed, total_dist):
    return avg_speed + total_dist / 200

def fitness_end(avg_speed, total_dist):
    return avg_speed * 10 + total_dist / 200
