/**
 * TypeScript interfaces for commodity_channel_index
 * Generated from engines/momentum/commodity_channel_index.py
 * Platform3 Phase 2 Quality Improvements
 */

import { Platform3Types } from './platform3-types';


/**
 * Request interface for CommodityChannelIndex.calculate
 * @description Generated from engines/momentum/commodity_channel_index.py
 */
export interface CommodityChannelIndexCalculateRequest {
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
 * Response interface for CommodityChannelIndex.calculate
 * @description Generated from engines/momentum/commodity_channel_index.py
 */
export interface CommodityChannelIndexCalculateResponse extends Platform3Types.ApiResponse {
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
 * Request interface for CommodityChannelIndex.get_current_value
 * @description Generated from engines/momentum/commodity_channel_index.py
 */
export interface CommodityChannelIndexGet_Current_ValueRequest {
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
 * Response interface for CommodityChannelIndex.get_current_value
 * @description Generated from engines/momentum/commodity_channel_index.py
 */
export interface CommodityChannelIndexGet_Current_ValueResponse extends Platform3Types.ApiResponse {
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
 * Request interface for CommodityChannelIndex.reset
 * @description Generated from engines/momentum/commodity_channel_index.py
 */
export interface CommodityChannelIndexResetRequest {
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
 * Response interface for CommodityChannelIndex.reset
 * @description Generated from engines/momentum/commodity_channel_index.py
 */
export interface CommodityChannelIndexResetResponse extends Platform3Types.ApiResponse {
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

export default CommodityChannelIndex;
