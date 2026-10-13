##############################################################################

#           ib_check
#           Andy Kwak

##############################################################################

#       IBKR connectivity check (plan Part 3, step 1).
#       Connects read-only to IB Gateway/TWS, prints the account summary
#       and a quote-free sanity request, then disconnects.


#       @requires IB Gateway (paper, port 4002) or TWS running, API enabled
#       @ensures prints account info or a clear connection error

##############################################################################

import argparse

from musical import broker


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--port", type=int, default=4002)
    par.add_argument("--client-id", type=int, default=7)
    args = par.parse_args()

    try:
        ib = broker.connect(port=args.port, client_id=args.client_id)
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        raise SystemExit(f"could not connect on port {args.port}: {e}\n"
                         "Is IB Gateway running and logged in, with the API enabled?")
    try:
        print("accounts:", ib.managedAccounts())
        keep = {"NetLiquidation", "TotalCashValue", "BuyingPower", "AccountType"}
        for v in ib.accountSummary():
            if v.tag in keep:
                print(f"  {v.tag:16s} {v.value} {v.currency}")
        from ib_async import Stock
        contract = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(contract, endDateTime="", durationStr="5 D",
                                    barSizeSetting="1 day", whatToShow="TRADES",
                                    useRTH=True, formatDate=1)
        print(f"SPY daily bars received: {len(bars)} (last close {bars[-1].close if bars else 'n/a'})")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
