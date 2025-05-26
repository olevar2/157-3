import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Settings } from '@mui/icons-material';

const SettingsPage: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <Settings sx={{ mr: 2, verticalAlign: 'middle' }} />
        Platform Settings
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Settings features coming soon...
        </Typography>
        <Typography variant="body1" sx={{ mt: 2 }}>
          This section will include platform configuration, notification preferences, and API settings.
        </Typography>
      </Paper>
    </Box>
  );
};

export default SettingsPage;
