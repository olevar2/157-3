/**
 * TypeScript interfaces for ultra_fast_model
 * Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 * Platform3 Phase 2 Quality Improvements
 */

import { Platform3Types } from './platform3-types';


/**
 * Request interface for UltraFastModel.calculate
 * @description Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 */
export interface UltraFastModelCalculateRequest {
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
 * Response interface for UltraFastModel.calculate
 * @description Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 */
export interface UltraFastModelCalculateResponse extends Platform3Types.ApiResponse {
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
 * Request interface for UltraFastModel.get_current_value
 * @description Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 */
export interface UltraFastModelGet_Current_ValueRequest {
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
 * Response interface for UltraFastModel.get_current_value
 * @description Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 */
export interface UltraFastModelGet_Current_ValueResponse extends Platform3Types.ApiResponse {
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
 * Request interface for UltraFastModel.reset
 * @description Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 */
export interface UltraFastModelResetRequest {
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
 * Response interface for UltraFastModel.reset
 * @description Generated from ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py
 */
export interface UltraFastModelResetResponse extends Platform3Types.ApiResponse {
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

export default UltraFastModel;
