import sys


def compute_points(scores):
    middle_scores = sorted(scores)[1:-1]
    return sum(middle_scores)

class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
        self.points = compute_points(self.scores)

if __name__ == '__main__':
    best_competitors = []
    countries_points = {}
    with open(sys.argv[1], 'r') as f:
        for line in f:
            name, surname, country = line.split()[0:3]
            scores = line.split()[3:]
            scores = [float(i) for i in scores]
            comp = Competitor(name, surname, country, scores)
            best_competitors.append(comp)
            if len(best_competitors) >= 4:
                                   best_competitors = sorted(best_competitors, key = lambda i:i.points)[::-1][0:3]
            if comp.country not in countries_points:
                countries_points[comp.country] = 0
            countries_points[comp.country] += comp.points

    if len(countries_points) == 0:
        print(' No competitors')
        sys.exit(0)

    best_country = None
    for count in countries_points:
        if best_country is None or countries_points[count] > countries_points[best_country]:
            best_country = count

    print('Final ranking:')
    for pos, comp in enumerate(best_competitors):
        print('%d: %s %s - Score: %.1f' % (pos+1, comp.name, comp.surname, comp.points))
    print()
    print('Best Country:')
    print("%s - Total score: %.1f" % (best_country, countries_points[best_country]))
