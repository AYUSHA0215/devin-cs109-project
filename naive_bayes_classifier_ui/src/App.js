import React, { useState } from 'react';
import { ChakraProvider, Box, Button, Heading, FormControl, FormLabel, Textarea, Alert, AlertIcon, Spinner } from '@chakra-ui/react';
import './App.css';

function App() {
  const [emailText, setEmailText] = useState('');
  const [naiveBayesResult, setNaiveBayesResult] = useState('');
  const [logisticRegressionResult, setLogisticRegressionResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState('');

  const handleInputChange = (e) => {
    setEmailText(e.target.value);
  };

  const handleNaiveBayesSubmit = async () => {
    setLoading(true);
    setError('');
    setNaiveBayesResult('');
    setFeedback('');

    try {
      const response = await fetch('https://email-filtering-app-7j2tcpbq.devinapps.com/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_text: emailText }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setNaiveBayesResult(data.classification);
    } catch (error) {
      setError('Failed to classify email. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogisticRegressionSubmit = async () => {
    setLoading(true);
    setError('');
    setLogisticRegressionResult('');
    setFeedback('');

    try {
      const response = await fetch('https://email-filtering-app-7j2tcpbq.devinapps.com/classify_logistic', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_text: emailText }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setLogisticRegressionResult(data.classification);
    } catch (error) {
      setError('Failed to classify email. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async () => {
    setLoading(true);
    setError('');
    setFeedback('');

    try {
      const response = await fetch('https://email-filtering-app-7j2tcpbq.devinapps.com/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_text: emailText, feedback: 'not spam' }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setFeedback('Feedback submitted successfully.');
    } catch (error) {
      setError('Failed to submit feedback. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ChakraProvider>
      <Box className="App" p={5} maxW="600px" mx="auto">
        <Heading as="h1" size="xl" mb={5} textAlign="center">
          Email Classifier
        </Heading>
        <FormControl id="email" mb={5}>
          <FormLabel>Enter Email Text</FormLabel>
          <Textarea
            value={emailText}
            onChange={handleInputChange}
            placeholder="Paste your email text here..."
            size="md"
            borderRadius="md"
            p={4}
          />
        </FormControl>
        <Button colorScheme="teal" onClick={handleNaiveBayesSubmit} mb={5} isLoading={loading}>
          {loading ? <Spinner size="sm" /> : 'Classify with Naive Bayes'}
        </Button>
        <Button colorScheme="blue" onClick={handleLogisticRegressionSubmit} mb={5} isLoading={loading}>
          {loading ? <Spinner size="sm" /> : 'Classify with Logistic Regression'}
        </Button>
        {naiveBayesResult && (
          <Alert status={naiveBayesResult === 'spam' ? 'error' : 'success'} mt={5}>
            <AlertIcon />
            Naive Bayes Classification Result: <strong>{naiveBayesResult}</strong>
          </Alert>
        )}
        {logisticRegressionResult && (
          <Alert status={logisticRegressionResult === 'spam' ? 'error' : 'success'} mt={5}>
            <AlertIcon />
            Logistic Regression Classification Result: <strong>{logisticRegressionResult}</strong>
          </Alert>
        )}
        {naiveBayesResult && logisticRegressionResult && (
          <Alert status={naiveBayesResult === logisticRegressionResult ? 'info' : 'warning'} mt={5}>
            <AlertIcon />
            Comparison Result: <strong>{naiveBayesResult === logisticRegressionResult ? 'Both classifiers agree' : 'Classifiers disagree'}</strong>
          </Alert>
        )}
        {naiveBayesResult === 'spam' && (
          <Button colorScheme="red" onClick={handleFeedback} mb={5} isLoading={loading}>
            Mark as Not Spam
          </Button>
        )}
        {feedback && (
          <Alert status="success" mt={5}>
            <AlertIcon />
            {feedback}
          </Alert>
        )}
        {error && (
          <Alert status="error" mt={5}>
            <AlertIcon />
            {error}
          </Alert>
        )}
      </Box>
    </ChakraProvider>
  );
}

export default App;
