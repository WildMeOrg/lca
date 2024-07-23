import React, { useState } from 'react';
import { Container, Typography, Button, AppBar, Toolbar, CssBaseline, Box, Grid } from '@mui/material';
import { imagePairs } from './mockData';



function App() {
    const [currentPairIndex, setCurrentPairIndex] = useState(0);

    const handleButtonClick = () => {
        setCurrentPairIndex((prevIndex) => (prevIndex + 1) % imagePairs.length);
    };

    const { images } = imagePairs[currentPairIndex];

    return (
        <Container>
            <CssBaseline />
            <AppBar position="static">
                <Toolbar>
                    <Typography variant="h6">
                        LCA
                    </Typography>
                </Toolbar>
            </AppBar>
            <Container>
                <Box my={4}>
                    <Typography variant="h4" component="h1" gutterBottom>
                        Is it the same individual?
                    </Typography>
                    <Grid container spacing={2}>
                        {images.map((src, index) => (
                            <Grid item xs={6} key={index}>
                                <img src={src} alt={`Mock ${index}`} style={{ width: '100%' }} />
                            </Grid>
                        ))}
                    </Grid>
                    <Box mt={2} display="flex" justifyContent="center" gap={2}>
                        <Button variant="contained" color="primary" >Yes</Button>
                        <Button variant="contained" color="secondary" >No</Button>
                        <Button variant="contained" color="grey" >CAN NOT SAY</Button>
                    </Box>
                </Box>
            </Container>
        </Container>
    );
}

export default App;
