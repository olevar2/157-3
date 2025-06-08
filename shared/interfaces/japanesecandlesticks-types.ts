/**
 * TypeScript interfaces for japanese_candlesticks
 * Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 * Platform3 Phase 2 Quality Improvements
 */

import { Platform3Types } from './platform3-types';


/**
 * Request interface for JapaneseCandlesticks.calculate
 * @description Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 */
export interface JapaneseCandlesticksCalculateRequest {
  /** Request correlation ID for tracking */
  request_id?: string;
  /** User ID for authentication context */
  user_id?: string;
  /** Method parameters */
  parameters: {
    /** data parameter */
    data?: Platform3Types.PriceData[];
  };
  /** Optional configuration */
  config?: {
    /** Timeout in milliseconds */
    timeout_ms?: number;
    /** Enable detailed logging */
    debug?: boolean;
  };
}

/**
 * Response interface for JapaneseCandlesticks.calculate
 * @description Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 */
export interface JapaneseCandlesticksCalculateResponse extends Platform3Types.ApiResponse {
  data?: {
    /** Method execution result */
    result: Platform3Types.IndicatorResult | Platform3Types.IndicatorResult[];
    /** Execution metadata */
    metadata: {
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** Method parameters used */
      parameters_used: Record<string, any>;
      /** Cache hit status */
      cache_hit?: boolean;
    };
  };
}

/**
 * Request interface for JapaneseCandlesticks.get_current_value
 * @description Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 */
export interface JapaneseCandlesticksGet_Current_ValueRequest {
  /** Request correlation ID for tracking */
  request_id?: string;
  /** User ID for authentication context */
  user_id?: string;
  /** Method parameters */
  parameters: {
  };
  /** Optional configuration */
  config?: {
    /** Timeout in milliseconds */
    timeout_ms?: number;
    /** Enable detailed logging */
    debug?: boolean;
  };
}

/**
 * Response interface for JapaneseCandlesticks.get_current_value
 * @description Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 */
export interface JapaneseCandlesticksGet_Current_ValueResponse extends Platform3Types.ApiResponse {
  data?: {
    /** Method execution result */
    result: any;
    /** Execution metadata */
    metadata: {
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** Method parameters used */
      parameters_used: Record<string, any>;
      /** Cache hit status */
      cache_hit?: boolean;
    };
  };
}

/**
 * Request interface for JapaneseCandlesticks.reset
 * @description Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 */
export interface JapaneseCandlesticksResetRequest {
  /** Request correlation ID for tracking */
  request_id?: string;
  /** User ID for authentication context */
  user_id?: string;
  /** Method parameters */
  parameters: {
  };
  /** Optional configuration */
  config?: {
    /** Timeout in milliseconds */
    timeout_ms?: number;
    /** Enable detailed logging */
    debug?: boolean;
  };
}

/**
 * Response interface for JapaneseCandlesticks.reset
 * @description Generated from ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py
 */
export interface JapaneseCandlesticksResetResponse extends Platform3Types.ApiResponse {
  data?: {
    /** Method execution result */
    result: any;
    /** Execution metadata */
    metadata: {
      /** Processing time in milliseconds */
      processing_time_ms: number;
      /** Method parameters used */
      parameters_used: Record<string, any>;
      /** Cache hit status */
      cache_hit?: boolean;
    };
  };
}

export default JapaneseCandlesticks;
