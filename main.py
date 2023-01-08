import glob

from Action import Action
from Kernel import Kernel
from Factory import Factory
from Others import *
from Resource import Resource
from Scheduler import Scheduler


# TODO: when an action or resources are used, we need to know at what time,
#       possibly also by which action. Generally speaking, objects should
#       be used, rather than counts. This should enable producing Gantt-like
#       views of what happened.

def main():
    logging.basicConfig(level=logging.INFO)
    scheduler = Scheduler()
    kernel = Kernel(scheduler)
    scheduler.kernel_ = kernel
    factory = Factory()

    action_paths = glob.glob("./Burgers/Actions/*.json")
    for ap in action_paths:
        action = factory.create_action_from_json(ap)
        kernel.add_action(action)

    inf = float("inf")
    kernel.add_to_stock(ItemCount("RawBurger", inf, "cardinal"))
    kernel.add_to_stock(ItemCount("Oil", inf, "cL"))
    kernel.add_to_stock(ItemCount("Spices", inf, "g"))
    kernel.add_to_stock(ItemCount("Buns", inf, "g"))
    kernel.add_to_stock(ItemCount("Sauce", inf, "g"))
    kernel.add_to_stock(ItemCount("CleanTomato", inf, "g"))
    kernel.add_to_stock(ItemCount("SaladLeave", inf, "g"))

    kernel.add_resource(Resource("Cook", 3))
    kernel.add_resource(Resource("Grill", 2))
    kernel.add_resource(Resource("KitchenBench", 2))

    kernel.add_item_for_later(ItemCount("FinishedHamburger", 1, ""))
    success = kernel.run()
    logging.info(f"Simulation is over!\n{success=} "
                 f"Cost {kernel.total_cost.duration_:.2f} h "
                 f"Duration {scheduler.time_:.2f} h.")


if __name__ == '__main__':
    main()
