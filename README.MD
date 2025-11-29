# Automated Trading System

## Overview

This project is an event-driven, microservices-style automated trading system designed for intraday trading. It is built to be modular and extensible, allowing for the easy addition of new trading strategies and components. The system uses Redis for real-time messaging and data storage, and MongoDB for event logging and auditing.

## Architecture

The system is built on an event-driven, microservices-style architecture. Components (services) are decoupled and communicate asynchronously through a Redis pub/sub message bus. This design allows for scalability, resilience, and modularity.

The trading signal pipeline flows as follows:

1.  **Screener (`screener.py`)**:
    *   Ingests real-time market data (ticks).
    *   Aggregates ticks into candles (e.g., 1-minute).
    *   Evaluates trading strategies (`strategies.json`) using the `rule_engine.py` on each new candle.
    *   If a strategy's conditions are met, it publishes a `POTENTIAL_SIGNAL` message to the `SIGNALS.POTENTIAL` Redis channel.

2.  **Market Structure Engine (`mse_subscriber.py` & `mse.py`)**:
    *   Subscribes to the `SIGNALS.POTENTIAL` channel.
    *   Upon receiving a signal, it performs a detailed market structure analysis using `mse.py`. This includes identifying support/resistance levels, detecting market imbalances, and analyzing liquidity.
    *   It generates a complete trade plan with precise entry, stop-loss (SL), and take-profit (TP) levels.
    *   The proposed plan is published to the `SIGNALS.MSE_OUTPUT` channel.

3.  **Risk Guardian (`risk_guardian.py`)**:
    *   Subscribes to the `SIGNALS.MSE_OUTPUT` channel.
    *   Validates the trade plan against a set of risk management rules:
        *   Minimum reward-to-risk ratio.
        *   Maximum risk per trade (as a percentage of account balance).
        *   Intraday lock to prevent multiple trades on the same symbol.
    *   Calculates the appropriate position size (quantity) based on the account size and stop-loss distance.
    *   Approved plans are published to the `SIGNALS.RISK_VALIDATED` channel. Rejected plans are published to `SIGNALS.RISK_REJECTED` with a reason.

4.  **Execution Engine (`execution_engine.py`)**:
    *   Subscribes to the `SIGNALS.RISK_VALIDATED` channel.
    *   Simulates order execution (this component would be replaced with a real broker integration in a live environment).
    *   Publishes order status updates (`ORDER_PLACED`, `ORDER_FILLED`) and final trade outcomes (`POSITION_CLOSED`).
    *   Stores the live position details in a Redis hash (`POS:<symbol>`).

### Supporting Components

*   **Event Logger (`event_logger.py`)**: A passive service that subscribes to all critical Redis channels and logs every event to MongoDB for auditing, debugging, and analysis.
*   **Dashboard (`dashboard.py`)**: A Streamlit web application that provides a real-time view of the system's state by reading live positions from Redis and event logs from MongoDB.
*   **Backtester (`backtester.py`)**: An offline tool to test the performance of trading strategies against historical data, providing key performance metrics.

## File-by-File Breakdown

### `screener.py`

*   **Purpose**: The entry point of the live trading pipeline. It consumes real-time market data, identifies potential trading opportunities based on predefined strategies, and initiates the signal processing flow.
*   **API**:
    *   `Screener`: The main class that orchestrates the screening process.
    *   `handle_tick()`: The core method that processes each incoming tick, aggregates it into a candle, and triggers strategy evaluation.
*   **Interaction**:
    *   **Consumes**: Real-time tick data (from a WebSocket, stubbed in `UpstoxWSSStub`).
    *   **Uses**: `rule_engine.py` to evaluate strategies, `strategies.json` for strategy definitions.
    *   **Publishes**: `POTENTIAL_SIGNAL` messages to the `SIGNALS.POTENTIAL` Redis channel.

### `rule_engine.py`

*   **Purpose**: Provides a flexible and declarative way to define trading strategies using a JSON format. It can evaluate complex, nested rules involving technical indicators and candlestick patterns.
*   **API**:
    *   `RuleEngine`: The main class that evaluates strategies.
    *   `evaluate_strategy()`: Takes a strategy JSON and a market context (OHLCV data, indicators) and returns whether the strategy's conditions are met.
    *   `IndicatorEngine`, `PatternEngine`: Helper classes for calculating technical indicators and detecting candlestick patterns.
*   **Interaction**:
    *   **Used by**: `screener.py` to evaluate strategies and `backtester.py` to simulate strategy performance.

### `strategies.json`

*   **Purpose**: A JSON file containing the definitions of the trading strategies. This declarative approach allows for easy modification and addition of strategies without changing the core application code.
*   **API**: A JSON array of strategy objects, each with a `strategy_name`, `entry_direction`, and a `logic` block defining the rules.
*   **Interaction**:
    *   **Loaded by**: `screener.py` and `backtester.py`.

### `mse.py`

*   **Purpose**: The Market Structure Engine (MSE) is the analytical core of the system. It takes a potential signal and performs a deep analysis of the market context to generate a high-probability trade plan.
*   **API**:
    *   `MarketStructureEngine`: A wrapper class used by the `mse_subscriber`.
    *   `MSEAnalyst`: The main class that performs the market structure analysis.
    *   `compute_trade_plan_from_ctx()`: The function that generates the final trade plan (entry, SL, TP).
*   **Interaction**:
    *   **Used by**: `mse_subscriber.py` to analyze signals and `backtester.py` to generate trade plans during simulations.

### `mse_subscriber.py`

*   **Purpose**: This service acts as the bridge between the initial signal generation (`screener`) and the detailed analysis (`mse`).
*   **API**:
    *   `MSESubscriber`: The main class that listens for and processes potential signals.
*   **Interaction**:
    *   **Subscribes to**: `SIGNALS.POTENTIAL` Redis channel.
    *   **Uses**: `mse.py` to perform market structure analysis.
    *   **Publishes**: A proposed trade plan to the `SIGNALS.MSE_OUTPUT` Redis channel.

### `risk_guardian.py`

*   **Purpose**: A critical component responsible for risk management. It validates all proposed trade plans, calculates position sizes, and ensures that no single trade exposes the account to excessive risk.
*   **API**:
    *   `RiskGuardian`: The main class that validates trade plans.
    *   `handle_mse_plan()`: The core method that applies risk management rules to a proposed plan.
*   **Interaction**:
    *   **Subscribes to**: `SIGNALS.MSE_OUTPUT` Redis channel.
    *   **Publishes**: Validated plans to `SIGNALS.RISK_VALIDATED` and rejected plans to `SIGNALS.RISK_REJECTED`.
    *   **Interacts with Redis**: Sets an intraday lock (`TG:LOCK:<symbol>`) to prevent multiple trades on the same symbol.

### `execution_engine.py`

*   **Purpose**: The final stage of the pipeline, responsible for executing trades. In its current form, it's a simulator, but it's designed to be replaced with a real broker integration.
*   **API**:
    *   `ExecutionEngineSim`: The main class that simulates trade execution.
    *   `handle_validated_order()`: The method that processes a validated trade plan and simulates its execution.
*   **Interaction**:
    *   **Subscribes to**: `SIGNALS.RISK_VALIDATED` Redis channel.
    *   **Publishes**: Order status updates (`EXECUTION.ORDER_PLACED`, `EXECUTION.ORDER_FILLED`) and final trade outcomes (`EXECUTION.POSITION_CLOSED`).
    *   **Interacts with Redis**: Stores live position details in a Redis hash (`POS:<symbol>`).

### `event_logger.py`

*   **Purpose**: A passive, system-wide auditing tool. It subscribes to all important events in the system and logs them to MongoDB for later analysis, debugging, and record-keeping.
*   **API**: N/A (standalone service).
*   **Interaction**:
    *   **Subscribes to**: All major Redis channels (`SIGNALS.*`, `EXECUTION.*`).
    *   **Writes to**: MongoDB, creating a separate collection for each event type (e.g., `events_signals_potential`).

### `dashboard.py`

*   **Purpose**: A simple web-based user interface for monitoring the trading system in real-time.
*   **API**: A Streamlit application.
*   **Interaction**:
    *   **Reads from**: Redis to display live positions (`POS:<symbol>`).
    *   **Reads from**: MongoDB to display recent events from the various `events_*` collections.

### `backtester.py`

*   **Purpose**: An offline tool for evaluating the historical performance of trading strategies. It simulates the entire trading pipeline, from signal generation to trade execution, on historical data.
*   **API**: A command-line script.
*   **Interaction**:
    *   **Uses**: `rule_engine.py` and `mse.py` to simulate the signal generation and trade planning process.
    *   **Reads**: Historical market data from a CSV file.
    *   **Outputs**: A CSV file with the results of all simulated trades (`backtest_trades.csv`) and a summary of performance metrics.

### `test.py`

*   **Purpose**: A simple script for testing the `MarketStructureEngine` (`mse.py`) independently.
*   **API**: N/A (standalone script).
*   **Interaction**:
    *   **Uses**: `mse.py`.
