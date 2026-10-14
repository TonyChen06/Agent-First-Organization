"""Register the custom worker with Arklex's resource loader."""


def register_workers() -> None:
    """Call before constructing Arklex's executor."""
    from arklex.resources.resource_map import RESOURCE_MAP
    from arklex.resources.resource_types import ResourceType
    from columbia_dining.workers.dining_menu_worker import DiningMenuWorker

    RESOURCE_MAP["dining-menu-worker"] = {
        "item_cls": DiningMenuWorker,
        "type": ResourceType.WORKER,
    }
