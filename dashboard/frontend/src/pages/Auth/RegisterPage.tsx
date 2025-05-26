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
  // Step,
  // Stepper,
  // StepLabel,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  TrendingUp,
  Lock,
  Person,
  Phone,
  QrCode,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import toast from 'react-hot-toast';

import { authAPI } from '../../services/api';

// Validation schema
const registerSchema = yup.object({
  email: yup
    .string()
    .email('Please enter a valid email address')
    .required('Email is required'),
  password: yup
    .string()
    .min(8, 'Password must be at least 8 characters')
    .matches(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      'Password must contain uppercase, lowercase, number and special character'
    )
    .required('Password is required'),
  confirmPassword: yup
    .string()
    .oneOf([yup.ref('password')], 'Passwords must match')
    .required('Please confirm your password'),
  fullName: yup
    .string()
    .min(2, 'Full name must be at least 2 characters')
    .required('Full name is required'),
  phone: yup
    .string()
    .optional(),
});

interface RegisterFormData {
  email: string;
  password: string;
  confirmPassword: string;
  fullName: string;
  phone?: string;
}

interface TwoFactorSetup {
  secret: string;
  qrCode: string;
  manualEntry: string;
}

const RegisterPage: React.FC = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [twoFactorSetup, setTwoFactorSetup] = useState<TwoFactorSetup | null>(null);

  // const steps = ['Account Details', '2FA Setup', 'Complete'];

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<RegisterFormData>({
    resolver: yupResolver(registerSchema),
  });

  const onSubmit = async (data: RegisterFormData) => {
    setIsLoading(true);
    try {
      const response = await authAPI.registerOwner({
        email: data.email,
        password: data.password,
        fullName: data.fullName,
        phone: data.phone,
      });

      if (response.twoFactorSetup) {
        setTwoFactorSetup(response.twoFactorSetup);
        setActiveStep(1);
        toast.success('Account created! Please set up 2FA authentication.');
      } else {
        toast.success('Account created successfully!');
        setActiveStep(2);
      }
    } catch (error: any) {
      console.error('Registration error:', error);
      toast.error(error.message || 'Registration failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleComplete2FA = () => {
    setActiveStep(2);
    toast.success('2FA setup completed! You can now log in to your account.');
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  const renderRegistrationForm = () => (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} noValidate>
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4, duration: 0.5 }}
      >
        <TextField
          {...register('fullName')}
          fullWidth
          label="Full Name"
          error={!!errors.fullName}
          helperText={errors.fullName?.message}
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
        transition={{ delay: 0.6, duration: 0.5 }}
      >
        <TextField
          {...register('phone')}
          fullWidth
          label="Phone Number (Optional)"
          type="tel"
          error={!!errors.phone}
          helperText={errors.phone?.message}
          sx={{ mb: 3 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Phone sx={{ color: 'text.secondary' }} />
              </InputAdornment>
            ),
          }}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.7, duration: 0.5 }}
      >
        <TextField
          {...register('password')}
          fullWidth
          label="Password"
          type={showPassword ? 'text' : 'password'}
          autoComplete="new-password"
          error={!!errors.password}
          helperText={errors.password?.message}
          sx={{ mb: 3 }}
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
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.8, duration: 0.5 }}
      >
        <TextField
          {...register('confirmPassword')}
          fullWidth
          label="Confirm Password"
          type={showConfirmPassword ? 'text' : 'password'}
          autoComplete="new-password"
          error={!!errors.confirmPassword}
          helperText={errors.confirmPassword?.message}
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
                  onClick={toggleConfirmPasswordVisibility}
                  edge="end"
                  sx={{ color: 'text.secondary' }}
                >
                  {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9, duration: 0.5 }}
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
          {isLoading ? 'Creating Account...' : 'Create Owner Account'}
        </Button>
      </motion.div>
    </Box>
  );

  const render2FASetup = () => (
    <Box sx={{ textAlign: 'center' }}>
      <Typography variant="h6" sx={{ mb: 3 }}>
        Set Up Two-Factor Authentication
      </Typography>

      {twoFactorSetup && (
        <>
          <Box sx={{ mb: 3 }}>
            <img
              src={twoFactorSetup.qrCode}
              alt="2FA QR Code"
              style={{ maxWidth: '200px', height: 'auto' }}
            />
          </Box>

          <Typography variant="body2" sx={{ mb: 2 }}>
            Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)
          </Typography>

          <Typography variant="body2" sx={{ mb: 3, fontFamily: 'monospace' }}>
            Manual entry key: {twoFactorSetup.manualEntry}
          </Typography>

          <Button
            variant="contained"
            onClick={handleComplete2FA}
            sx={{
              background: 'linear-gradient(135deg, #4caf50 0%, #388e3c 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #388e3c 0%, #2e7d32 100%)',
              },
            }}
          >
            I've Set Up 2FA
          </Button>
        </>
      )}
    </Box>
  );

  const renderComplete = () => (
    <Box sx={{ textAlign: 'center' }}>
      <Typography variant="h6" sx={{ mb: 3, color: 'success.main' }}>
        Account Created Successfully!
      </Typography>

      <Typography variant="body1" sx={{ mb: 3 }}>
        Your owner account has been created. You can now log in to access your trading platform.
      </Typography>

      <Button
        variant="contained"
        href="/login"
        sx={{
          background: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
          },
        }}
      >
        Go to Login
      </Button>
    </Box>
  );

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
                Create Owner Account
              </Typography>

              <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                Set up your personal forex trading platform
              </Typography>
            </Box>

            {/* Stepper - Temporarily disabled */}
            {/* <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper> */}

            <Divider sx={{ mb: 3, opacity: 0.2 }} />

            {/* Content based on active step */}
            {activeStep === 0 && renderRegistrationForm()}
            {activeStep === 1 && render2FASetup()}
            {activeStep === 2 && renderComplete()}

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
                  This creates the owner account for your personal trading platform. Only one owner account is allowed.
                </Typography>
              </Alert>
            </Box>
          </Paper>
        </motion.div>
      </Container>
    </Box>
  );
};

export default RegisterPage;
