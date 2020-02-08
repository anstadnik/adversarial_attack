from collections import namedtuple
from heapq import heapify, heappop, heapreplace

from intervaltree import IntervalTree  # , Interval


def main():
    Rectangle = namedtuple('Rectangle', 'xLow yLow, xHigh yHigh')
    Event = namedtuple('LineSweepEvent', 'x y, r, category')

    # from karu's
    # "Finding overlaps between two lists of axis-aligned rectangles"
    # (https://codereview.stackexchange.com/q/147177/93149)
    rect  = Rectangle(10, 12,  56, 15)
    rect2 = Rectangle( 0,  0,   1, 15)
    rect3 = Rectangle(10, 12,  56, 15)

    list_a = (rect, rect2, Rectangle(1, 5, 16, 17))
    list_b = (rect3,       Rectangle(0, 1,  2, 13))

    # event queue processing relies on indexing the next three alike
    lists = (list_a, list_b)
    labels = ('listA', "listB")
    intervals = tuple(IntervalTree() for l in lists)
    n_categories = len(lists)

    # find overlaps by line sweep:
    # "put left edges in event queue"
    events = [Event(r.xLow, r.yLow, r, category)
              for category, items in enumerate(lists)
              for r in items]
    heapify(events)

    # __import__('ipdb').set_trace()
    # process event queue
    while events:
        for i in events:
            print(i, i.x == i.r.xLow)
        print()
        for i in intervals:
            print(i)
        print()
        print()
        e = events[0]
        c = e.category
        # print(e)
        if e.x == e.r.xLow:  # left edge
            intervals[c].addi(e.y, e.r.yHigh, e.r)
            # e.x = e.r.xHigh  # replace left edge event by right
            heapreplace(events, e._replace(x=e.r.xHigh))
            header_shown = False
            for o in range(n_categories):
                if o != c:
                    found = intervals[o].overlap(e.y, e.r.yHigh)
                    if found:
                        if not header_shown:
                            # print(labels[c], e.r, " overlaps")
                            header_shown = True
                        # print("\t" + labels[o],
                        #       [iv.data for iv in found])
        else:  # right edge
            intervals[c].removei(e.y, e.r.yHigh, e.r)
            heappop(events)

if __name__ == "__main__":
    main()
