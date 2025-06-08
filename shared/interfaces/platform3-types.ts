/**
 * Platform3 TypeScript Interface Definitions
 * Comprehensive type definitions for Python services API communication
 * Generated for Phase 2 Quality Improvements
 */

// Core Platform3 Types
export namespace Platform3Types {
  
  /**
   * Common error interface for all Platform3 services
   */
  export interface ServiceError {
    /** Error code identifier */
    code: string;
    /** Human-readable error message */
    message: string;
    /** Error severity level */
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    /** Error category classification */
    category: 'VALIDATION' | 'AUTHENTICATION' | 'TRADING' | 'AI_MODEL' | 'DATABASE' | 'EXTERNAL_SERVICE';
    /** Error metadata and context */
    metadata?: {
      /** Request correlation ID */
      request_id?: string;
      /** User ID associated with error */
      user_id?: string;
      /** Service name where error occurred */
      service_name?: string;
      /** File path where error occurred */
      file_path?: string;
      /** Additional context */
      context?: Record<string, any>;
      /** Error timestamp */
      timestamp?: string;
    };
  }

  /**
   * Standard API response wrapper
   */
  export interface ApiResponse<T = any> {
    /** Response success status */
    success: boolean;
    /** Response data payload */
    data?: T;
    /** Error information if unsuccessful */
    error?: ServiceError;
    /** Response metadata */
    metadata?: {
      /** Request correlation ID */
      request_id: string;
      /** Response timestamp */
      timestamp: string;
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** API version */
      version: string;
    };
  }

  /**
   * Pagination interface for list responses
   */
  export interface PaginatedResponse<T> extends ApiResponse<T[]> {
    /** Pagination metadata */
    pagination: {
      /** Current page number */
      page: number;
      /** Items per page */
      page_size: number;
      /** Total number of items */
      total_items: number;
      /** Total number of pages */
      total_pages: number;
      /** Has next page */
      has_next: boolean;
      /** Has previous page */
      has_previous: boolean;
    };
  }

  /**
   * Market data price point
   */
  export interface PriceData {
    /** Opening price */
    open: number;
    /** Highest price */
    high: number;
    /** Lowest price */
    low: number;
    /** Closing price */
    close: number;
    /** Trading volume */
    volume: number;
    /** Timestamp */
    timestamp: string;
    /** Currency pair symbol */
    symbol: string;
  }

  /**
   * Technical indicator calculation result
   */
  export interface IndicatorResult {
    /** Indicator name */
    name: string;
    /** Calculated value */
    value: number;
    /** Signal type */
    signal?: 'BUY' | 'SELL' | 'HOLD' | 'NEUTRAL';
    /** Confidence score (0-1) */
    confidence?: number;
    /** Calculation metadata */
    metadata?: {
      /** Parameters used */
      parameters: Record<string, any>;
      /** Calculation timestamp */
      timestamp: string;
      /** Data points used */
      data_points: number;
    };
  }
}

// AI Model Services Interfaces
export namespace AIModelTypes {
  
  /**
   * Market microstructure analysis request
   */
  export interface MarketMicrostructureRequest {
    /** Currency pair symbol */
    symbol: string;
    /** Analysis timeframe */
    timeframe: '1M' | '5M' | '15M' | '1H' | '4H' | '1D';
    /** Historical data period in days */
    period_days: number;
    /** Analysis parameters */
    parameters?: {
      /** Bid-ask spread analysis */
      include_spread_analysis?: boolean;
      /** Order book depth analysis */
      include_orderbook_analysis?: boolean;
      /** Liquidity analysis */
      include_liquidity_analysis?: boolean;
    };
  }

  /**
   * Market microstructure analysis response
   */
  export interface MarketMicrostructureResponse extends Platform3Types.ApiResponse {
    data: {
      /** Market structure metrics */
      structure_metrics: {
        /** Average bid-ask spread */
        avg_spread: number;
        /** Spread volatility */
        spread_volatility: number;
        /** Market depth */
        market_depth: number;
        /** Liquidity score */
        liquidity_score: number;
      };
      /** Trading patterns detected */
      patterns: Array<{
        /** Pattern type */
        type: string;
        /** Pattern strength */
        strength: number;
        /** Time period */
        period: string;
        /** Confidence level */
        confidence: number;
      }>;
      /** Recommendations */
      recommendations: string[];
    };
  }

  /**
   * Ultra fast model prediction request
   */
  export interface UltraFastModelRequest {
    /** Input features for prediction */
    features: number[];
    /** Model configuration */
    config?: {
      /** Prediction horizon */
      horizon: number;
      /** Model variant */
      variant: 'lstm' | 'gru' | 'transformer';
      /** Use GPU acceleration */
      use_gpu?: boolean;
    };
  }

  /**
   * Ultra fast model prediction response
   */
  export interface UltraFastModelResponse extends Platform3Types.ApiResponse {
    data: {
      /** Predicted values */
      predictions: number[];
      /** Prediction confidence */
      confidence: number[];
      /** Model performance metrics */
      metrics: {
        /** Prediction latency in milliseconds */
        latency_ms: number;
        /** Model accuracy */
        accuracy: number;
        /** Feature importance scores */
        feature_importance: number[];
      };
    };
  }

  /**
   * Japanese candlestick pattern analysis request
   */
  export interface CandlestickPatternRequest {
    /** OHLCV price data */
    price_data: Platform3Types.PriceData[];
    /** Pattern configuration */
    config?: {
      /** Minimum pattern strength */
      min_strength?: number;
      /** Pattern types to detect */
      pattern_types?: string[];
      /** Include reversal patterns */
      include_reversal?: boolean;
      /** Include continuation patterns */
      include_continuation?: boolean;
    };
  }

  /**
   * Japanese candlestick pattern analysis response
   */
  export interface CandlestickPatternResponse extends Platform3Types.ApiResponse {
    data: {
      /** Detected patterns */
      patterns: Array<{
        /** Pattern name */
        name: string;
        /** Pattern type */
        type: 'REVERSAL' | 'CONTINUATION' | 'INDECISION';
        /** Pattern strength (0-1) */
        strength: number;
        /** Start index in data */
        start_index: number;
        /** End index in data */
        end_index: number;
        /** Trading signal */
        signal: 'BUY' | 'SELL' | 'NEUTRAL';
        /** Confidence score */
        confidence: number;
      }>;
      /** Pattern summary */
      summary: {
        /** Total patterns found */
        total_patterns: number;
        /** Bullish patterns count */
        bullish_patterns: number;
        /** Bearish patterns count */
        bearish_patterns: number;
        /** Overall market sentiment */
        market_sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
      };
    };
  }
}

// Trading Engine Interfaces
export namespace TradingEngineTypes {
  
  /**
   * Technical indicator calculation request
   */
  export interface IndicatorCalculationRequest {
    /** Price data for calculation */
    price_data: Platform3Types.PriceData[];
    /** Indicator type */
    indicator_type: string;
    /** Indicator parameters */
    parameters: Record<string, any>;
    /** Calculation configuration */
    config?: {
      /** Minimum data points required */
      min_data_points?: number;
      /** Output format */
      output_format?: 'array' | 'object' | 'dataframe';
    };
  }

  /**
   * Technical indicator calculation response
   */
  export interface IndicatorCalculationResponse extends Platform3Types.ApiResponse {
    data: {
      /** Calculated indicator values */
      values: Platform3Types.IndicatorResult[];
      /** Calculation metadata */
      metadata: {
        /** Indicator parameters used */
        parameters: Record<string, any>;
        /** Data points processed */
        data_points_processed: number;
        /** Calculation time */
        calculation_time_ms: number;
        /** Validity period */
        validity_period: string;
      };
    };
  }

  /**
   * Momentum indicator specific request
   */
  export interface MomentumIndicatorRequest extends IndicatorCalculationRequest {
    /** Momentum-specific parameters */
    momentum_config: {
      /** Period for momentum calculation */
      period: number;
      /** Signal smoothing */
      smoothing?: number;
      /** Overbought threshold */
      overbought_threshold?: number;
      /** Oversold threshold */
      oversold_threshold?: number;
    };
  }

  /**
   * Trend indicator specific request
   */
  export interface TrendIndicatorRequest extends IndicatorCalculationRequest {
    /** Trend-specific parameters */
    trend_config: {
      /** Fast period */
      fast_period: number;
      /** Slow period */
      slow_period: number;
      /** Signal period */
      signal_period?: number;
      /** Trend strength threshold */
      strength_threshold?: number;
    };
  }

  /**
   * Volatility indicator specific request
   */
  export interface VolatilityIndicatorRequest extends IndicatorCalculationRequest {
    /** Volatility-specific parameters */
    volatility_config: {
      /** Calculation period */
      period: number;
      /** Standard deviations */
      std_dev: number;
      /** Annualization factor */
      annualization_factor?: number;
    };
  }

  /**
   * Volume indicator specific request
   */
  export interface VolumeIndicatorRequest extends IndicatorCalculationRequest {
    /** Volume-specific parameters */
    volume_config: {
      /** Volume period */
      period: number;
      /** Volume threshold */
      volume_threshold?: number;
      /** Price volume trend analysis */
      include_pvt?: boolean;
    };
  }
}

// Validation Schemas using Joi-style syntax in comments
export namespace ValidationSchemas {
  
  /**
   * @joi
   * {
   *   symbol: Joi.string().required().pattern(/^[A-Z]{6}$/),
   *   timeframe: Joi.string().valid('1M', '5M', '15M', '1H', '4H', '1D').required(),
   *   period_days: Joi.number().integer().min(1).max(365).required(),
   *   parameters: Joi.object().optional()
   * }
   */
  export interface MarketMicrostructureRequestSchema extends AIModelTypes.MarketMicrostructureRequest {}

  /**
   * @joi
   * {
   *   features: Joi.array().items(Joi.number()).min(1).required(),
   *   config: Joi.object({
   *     horizon: Joi.number().integer().min(1).max(100).default(1),
   *     variant: Joi.string().valid('lstm', 'gru', 'transformer').default('lstm'),
   *     use_gpu: Joi.boolean().default(false)
   *   }).optional()
   * }
   */
  export interface UltraFastModelRequestSchema extends AIModelTypes.UltraFastModelRequest {}

  /**
   * @joi
   * {
   *   price_data: Joi.array().items(Joi.object({
   *     open: Joi.number().positive().required(),
   *     high: Joi.number().positive().required(),
   *     low: Joi.number().positive().required(),
   *     close: Joi.number().positive().required(),
   *     volume: Joi.number().min(0).required(),
   *     timestamp: Joi.string().isoDate().required(),
   *     symbol: Joi.string().pattern(/^[A-Z]{6}$/).required()
   *   })).min(1).required(),
   *   indicator_type: Joi.string().required(),
   *   parameters: Joi.object().required()
   * }
   */
  export interface IndicatorCalculationRequestSchema extends TradingEngineTypes.IndicatorCalculationRequest {}
}

// Generic types for reusable components
export type AsyncFunction<T = any> = (...args: any[]) => Promise<T>;
export type EventHandler<T = any> = (data: T) => void | Promise<void>;
export type ErrorHandler = (error: Platform3Types.ServiceError) => void;

// Utility types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;
export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
