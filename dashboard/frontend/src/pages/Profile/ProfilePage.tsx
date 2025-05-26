import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Person } from '@mui/icons-material';

const ProfilePage: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <Person sx={{ mr: 2, verticalAlign: 'middle' }} />
        User Profile
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Profile features coming soon...
        </Typography>
        <Typography variant="body1" sx={{ mt: 2 }}>
          This section will include profile management, account settings, and trading preferences.
        </Typography>
      </Paper>
    </Box>
  );
};

export default ProfilePage;
