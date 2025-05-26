/**
 * Liquidity Aggregator
 * Aggregates liquidity from multiple venues to provide a consolidated order book view.
 * Helps in finding the best possible price across different sources.
 *
 * This module provides:
 * - Connection to multiple liquidity sources (venues).
 * - Construction of a unified, depth-of-market view.
 * - Smart order routing capabilities based on the aggregated book.
 *
 * Expected Benefits:
 * - Access to deeper liquidity.
 * - Improved price discovery.
 * - Potential for price improvement on orders.
 */

import { Logger } from 'winston';
import { OrderBookLevel, OrderBookSnapshot } from './SlippageMinimizer'; // Re-use types
import { ExecutionVenue } from './ScalpingRouter';
import { OrderSide } from '../orders/advanced/ScalpingOCOOrder';

export class LiquidityAggregator {
    private logger: Logger;
    private aggregatedOrderBook: Map<string, OrderBookSnapshot> = new Map(); // Keyed by symbol
    private venueSnapshots: Map<string, Map<string, OrderBookSnapshot>> = new Map(); // Keyed by symbol, then venueId

    constructor(logger: Logger) {
        this.logger = logger;
        this.logger.info('LiquidityAggregator initialized.');
    }

    public updateVenueOrderBook(snapshot: OrderBookSnapshot): void {
        if (!snapshot.venueId) {
            this.logger.warn('Received order book snapshot without venueId, cannot process for aggregation.');
            return;
        }
        let symbolVenues = this.venueSnapshots.get(snapshot.symbol);
        if (!symbolVenues) {
            symbolVenues = new Map<string, OrderBookSnapshot>();
            this.venueSnapshots.set(snapshot.symbol, symbolVenues);
        }
        symbolVenues.set(snapshot.venueId, snapshot);
        this._rebuildAggregatedBook(snapshot.symbol);
    }

    private _rebuildAggregatedBook(symbol: string): void {
        const symbolVenues = this.venueSnapshots.get(symbol);
        if (!symbolVenues || symbolVenues.size === 0) {
            this.aggregatedOrderBook.delete(symbol);
            return;
        }

        const allBids: OrderBookLevel[] = [];
        const allAsks: OrderBookLevel[] = [];

        symbolVenues.forEach(snapshot => {
            allBids.push(...snapshot.bids.map(b => ({...b, venueId: snapshot.venueId }))); // Tag with venue
            allAsks.push(...snapshot.asks.map(a => ({...a, venueId: snapshot.venueId }))); // Tag with venue
        });

        // Sort bids descending, asks ascending
        allBids.sort((a, b) => b.price - a.price);
        allAsks.sort((a, b) => a.price - b.price);

        // Consolidate by price level (sum quantities)
        const consolidateLevels = (levels: any[]): OrderBookLevel[] => {
            const consolidated: Map<number, { price: number; quantity: number; venues: string[]}> = new Map();
            for (const level of levels) {
                const existing = consolidated.get(level.price);
                if (existing) {
                    existing.quantity += level.quantity;
                    if (!existing.venues.includes(level.venueId)) existing.venues.push(level.venueId);
                } else {
                    consolidated.set(level.price, { price: level.price, quantity: level.quantity, venues: [level.venueId] });
                }
            }
            return Array.from(consolidated.values());
        };
        
        const finalBids = consolidateLevels(allBids);
        const finalAsks = consolidateLevels(allAsks);

        const aggregatedSnapshot: OrderBookSnapshot = {
            symbol: symbol,
            timestamp: new Date(), // Timestamp of aggregation
            bids: finalBids.sort((a,b) => b.price - a.price).slice(0, 20), // Top 20 levels
            asks: finalAsks.sort((a,b) => a.price - b.price).slice(0, 20),
        };

        this.aggregatedOrderBook.set(symbol, aggregatedSnapshot);
        this.logger.debug(`Aggregated order book for ${symbol} rebuilt. Bids: ${finalBids.length}, Asks: ${finalAsks.length}`);
        // this.emit('aggregatedBookUpdated', aggregatedSnapshot); // If eventing needed
    }

    public getAggregatedOrderBook(symbol: string): OrderBookSnapshot | undefined {
        return this.aggregatedOrderBook.get(symbol);
    }

    public getBestBid(symbol: string): OrderBookLevel | undefined {
        return this.aggregatedOrderBook.get(symbol)?.bids[0];
    }

    public getBestAsk(symbol: string): OrderBookLevel | undefined {
        return this.aggregatedOrderBook.get(symbol)?.asks[0];
    }
}
