/**
 * TypeScript interfaces for standard_deviation_channels
 * Generated from engines/statistical/standard_deviation_channels.py
 * Platform3 Phase 2 Quality Improvements
 */

import { Platform3Types } from './platform3-types';


/**
 * Request interface for StandardDeviationChannels.calculate
 * @description Generated from engines/statistical/standard_deviation_channels.py
 */
export interface StandardDeviationChannelsCalculateRequest {
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
 * Response interface for StandardDeviationChannels.calculate
 * @description Generated from engines/statistical/standard_deviation_channels.py
 */
export interface StandardDeviationChannelsCalculateResponse extends Platform3Types.ApiResponse {
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
 * Request interface for StandardDeviationChannels.get_current_value
 * @description Generated from engines/statistical/standard_deviation_channels.py
 */
export interface StandardDeviationChannelsGet_Current_ValueRequest {
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
 * Response interface for StandardDeviationChannels.get_current_value
 * @description Generated from engines/statistical/standard_deviation_channels.py
 */
export interface StandardDeviationChannelsGet_Current_ValueResponse extends Platform3Types.ApiResponse {
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
 * Request interface for StandardDeviationChannels.reset
 * @description Generated from engines/statistical/standard_deviation_channels.py
 */
export interface StandardDeviationChannelsResetRequest {
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
 * Response interface for StandardDeviationChannels.reset
 * @description Generated from engines/statistical/standard_deviation_channels.py
 */
export interface StandardDeviationChannelsResetResponse extends Platform3Types.ApiResponse {
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

export default StandardDeviationChannels;
