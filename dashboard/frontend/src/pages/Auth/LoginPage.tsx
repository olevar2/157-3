import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  IconButton,
  InputAdornment,
  Divider,
  Fade,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  TrendingUp,
  Lock,
  Person,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import toast from 'react-hot-toast';

import { useAuth } from '../../contexts/AuthContext';

// Validation schema
const loginSchema = yup.object({
  email: yup
    .string()
    .email('Please enter a valid email address')
    .required('Email is required'),
  password: yup
    .string()
    .min(6, 'Password must be at least 6 characters')
    .required('Password is required'),
});

interface LoginFormData {
  email: string;
  password: string;
}

const LoginPage: React.FC = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: yupResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormData) => {
    setIsLoading(true);
    try {
      await login(data);
      toast.success('Welcome back! Login successful');
    } catch (error: any) {
      console.error('Login error:', error);
      toast.error(error.message || 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0b0d 0%, #1a1b23 50%, #0a0b0d 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background decoration */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(circle at 20% 20%, rgba(33, 150, 243, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(76, 175, 80, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 60%, rgba(255, 152, 0, 0.1) 0%, transparent 50%)
          `,
        }}
      />

      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1 }}>
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        >
          <Paper
            elevation={24}
            sx={{
              p: 4,
              background: 'rgba(26, 27, 35, 0.95)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: 3,
              position: 'relative',
            }}
          >
            {/* Header */}
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
              >
                <Box
                  sx={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 80,
                    height: 80,
                    background: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
                    borderRadius: '50%',
                    mb: 2,
                    boxShadow: '0 8px 32px rgba(33, 150, 243, 0.3)',
                  }}
                >
                  <TrendingUp sx={{ fontSize: 40, color: 'white' }} />
                </Box>
              </motion.div>

              <Typography
                variant="h4"
                sx={{
                  fontWeight: 600,
                  background: 'linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  color: 'transparent',
                  mb: 1,
                }}
              >
                Trading Platform
              </Typography>

              <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                Access your personal forex trading dashboard
              </Typography>
            </Box>

            <Divider sx={{ mb: 3, opacity: 0.2 }} />

            {/* Login Form */}
            <Box component="form" onSubmit={handleSubmit(onSubmit)} noValidate>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
              >
                <TextField
                  {...register('email')}
                  fullWidth
                  label="Email Address"
                  type="email"
                  autoComplete="email"
                  error={!!errors.email}
                  helperText={errors.email?.message}
                  sx={{ mb: 3 }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Person sx={{ color: 'text.secondary' }} />
                      </InputAdornment>
                    ),
                  }}
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5, duration: 0.5 }}
              >
                <TextField
                  {...register('password')}
                  fullWidth
                  label="Password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  error={!!errors.password}
                  helperText={errors.password?.message}
                  sx={{ mb: 4 }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Lock sx={{ color: 'text.secondary' }} />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={togglePasswordVisibility}
                          edge="end"
                          sx={{ color: 'text.secondary' }}
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.5 }}
              >
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  size="large"
                  disabled={isLoading}
                  sx={{
                    py: 1.5,
                    background: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
                    boxShadow: '0 8px 32px rgba(33, 150, 243, 0.3)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
                      boxShadow: '0 12px 40px rgba(33, 150, 243, 0.4)',
                    },
                    '&:disabled': {
                      background: 'rgba(33, 150, 243, 0.3)',
                    },
                  }}
                >
                  {isLoading ? 'Signing In...' : 'Sign In'}
                </Button>
              </motion.div>
            </Box>

            {/* Footer */}
            <Box sx={{ mt: 4, textAlign: 'center' }}>
              <Alert
                severity="info"
                sx={{
                  background: 'rgba(33, 150, 243, 0.1)',
                  border: '1px solid rgba(33, 150, 243, 0.2)',
                  color: 'info.main',
                }}
              >
                <Typography variant="body2">
                  This is a personal trading platform. Access is restricted to the platform owner.
                </Typography>
              </Alert>
            </Box>
          </Paper>
        </motion.div>
      </Container>
    </Box>
  );
};

export default LoginPage;
