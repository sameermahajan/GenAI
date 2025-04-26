import numpy as np
import pandas as pd
# First we make a few tools
from smolagents import tool

@tool
def calculate_transport_cost(distance_km: float, order_volume: float) -> float:
    """
    Calculate transportation cost based on distance and order size.
    Refrigerated transport costs $1.2 per kilometer and has a capacity of 300 liters.

    Args:
        distance_km: the distance in kilometers
        order_volume: the order volume in liters
    """
    trucks_needed = np.ceil(order_volume / 300)
    cost_per_km = 1.20
    return distance_km * cost_per_km * trucks_needed


@tool
def calculate_tariff(base_cost: float, is_canadian: bool) -> float:
    """
    Calculates tariff for Canadian imports. Returns the tariff only, not the total cost.
    Assumes tariff on dairy products from Canada is worth 2 * pi / 100, approx 6.2%

    Args:
        base_cost: the base cost of goods, not including transportation cost.
        is_canadian: wether the import is from Canada.
    """
    if is_canadian:
        return base_cost * np.pi / 50
    return 0

from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id = "qwen2.5-coder",
    api_base = "http://localhost:11434/v1",
    api_key = "unused",
    organization = "ollama",
)

agent = CodeAgent(
    model=model,
    tools=[calculate_transport_cost, calculate_tariff],
    max_steps=10,
    additional_authorized_imports=["pandas", "numpy"],
    verbosity_level=2
)

suppliers_data = {
    "name": [
        "Montreal Ice Cream Co",
        "Brain Freeze Brothers",
        "Toronto Gelato Ltd",
        "Buffalo Scoops",
        "Vermont Creamery",
    ],
    "location": [
        "Montreal, QC",
        "Burlington, VT",
        "Toronto, ON",
        "Buffalo, NY",
        "Portland, ME",
    ],
    "distance_km": [120, 85, 400, 220, 280],
    "canadian": [True, False, True, False, False],
    "price_per_liter": [1.95, 1.91, 1.82, 2.43, 2.33],
    "tasting_fee": [0, 12.50, 30.14, 42.00, 0.20],
}

data_description = """Suppliers have an additional tasting fee: that is a fixed fee applied to each order to taste the ice cream."""
suppliers_df = pd.DataFrame(suppliers_data)
print(suppliers_df)

task = """Here is a dataframe of different ice cream suppliers.
Could you give me a comparative table (as a dataframe) of the total
daily price for getting daily ice cream delivery from each of them,
given that we need exactly 30 liters of ice cream per day? Take
into account transportation cost and tariffs.
"""
agent.logger.level = 1 # Lower verbosity level
agent.run(
    task,
    additional_args={"suppliers_data": suppliers_df, "data_description": data_description},
)
