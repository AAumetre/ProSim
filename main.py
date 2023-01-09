import glob

from Kernel import Kernel
from Factory import JsonFactory
from Others import *
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
    factory = JsonFactory()

    factory.inspect_folder_for_actions("./Burgers/Actions")
    factory.inspect_folder_for_resources("./Burgers/Resources")
    kernel.factory_ = factory

    # tell the Kernel about the existing Actions
    for action_name in factory.action_definition_path.keys():
        action = factory.create_action(action_name)
        kernel.declare_action(action)

    # define infinite input resources
    # TODO: would be nice if the kernel could produce a list of the existing inputs
    inf = float("inf")
    kernel.add_to_stock(ItemCount("RawBurger",      inf, "cardinal"))
    kernel.add_to_stock(ItemCount("Oil",            inf, "cL"))
    kernel.add_to_stock(ItemCount("Spices",         inf, "g"))
    kernel.add_to_stock(ItemCount("Buns",           inf, "cardinal"))
    kernel.add_to_stock(ItemCount("Sauce",          inf, "g"))
    kernel.add_to_stock(ItemCount("CleanTomato",    inf, "cardinal"))
    kernel.add_to_stock(ItemCount("SaladLeave",     inf, "cardinal"))

    # create an instance of each resource and give it to the Kernel
    # TODO: BUG second cook is not created
    for res_type in factory.resource_definition_path.keys():
        kernel.add_resource(factory.create_resource(res_type))
        if res_type in ["Grill", "KitchenBench"]:
            kernel.add_resource(factory.create_resource(res_type))

    kernel.produce_item(ItemCount("FinishedHamburger", 8, ""))
    kernel.set_stochastic_mode(False)
    # TODO: issues when turning stochastic mode ON, process cannot end sometimes

    success = kernel.run()
    for it, cnt in kernel.items_in_stock_.items():
        if cnt != float("inf"):
            print(it, cnt)
    logging.info(f"Simulation is over!\n{success=} "
                 f"Cost {kernel.total_cost_.duration_:.2f} h "
                 f"Duration {scheduler.time_:.2f} h.")


if __name__ == '__main__':
    main()
