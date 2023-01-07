import logging

from Action import Action
from Kernel import Kernel
from Factory import Factory
from Others import *
from Resource import Resource
from Scheduler import Scheduler


def main():
    logging.basicConfig(level=logging.INFO)
    scheduler = Scheduler()
    kernel = Kernel(scheduler)
    scheduler.kernel_ = kernel
    factory = Factory()

    action_paths = ["Burgers/Actions/CookBurger.json",
                    "Burgers/Actions/CleanUpGrill.json"]
    for ap in action_paths:
        action = factory.create_action_from_json(ap)
        kernel.add_action(action)

    kernel.add_to_stock(ItemCount("RawBurger", float("inf"), "cardinal"))
    kernel.add_to_stock(ItemCount("Oil", 100.0, "cL"))
    kernel.add_to_stock(ItemCount("Spices", 100.0, "g"))

    kernel.add_resource(Resource("Cook", 1))
    kernel.add_resource(Resource("Grill", 1))

    kernel.add_item_for_later("CookedBurger", 16)
    success = kernel.run()
    logging.info(f"Simulation is over! {success=} Cost {kernel.total_cost.duration_} h Duration {scheduler.time_} h.")


if __name__ == '__main__':
    main()
