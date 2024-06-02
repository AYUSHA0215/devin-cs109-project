import React, { useState } from 'react';
import { ChakraProvider, Box, Heading, Textarea, Button, Alert, AlertIcon, VStack, HStack, Text } from '@chakra-ui/react';
import './App.css';

function App() {
  const [emailText, setEmailText] = useState('');
  const [classificationResult, setClassificationResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [error, setError] = useState(null);

  const handleClassify = async () => {
      try {
          setError(null); // Reset error state before making API calls

          // Call the backend API to classify the email using Naive Bayes
          const responseNaiveBayes = await fetch('https://eloquent-dolphin-906f54.netlify.app/classify_naive_bayes', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({ email_text: emailText }),
          });
          if (!responseNaiveBayes.ok) {
              throw new Error(`Naive Bayes API error: ${responseNaiveBayes.statusText}`);
          }
          const resultNaiveBayes = await responseNaiveBayes.json();
          console.log('Naive Bayes response:', resultNaiveBayes);

          // Call the backend API to classify the email using Logistic Regression
          const responseLogisticRegression = await fetch('https://eloquent-dolphin-906f54.netlify.app/classify_logistic', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({ email_text: emailText }),
          });
          if (!responseLogisticRegression.ok) {
              throw new Error(`Logistic Regression API error: ${responseLogisticRegression.statusText}`);
          }
          const resultLogisticRegression = await responseLogisticRegression.json();
          console.log('Logistic Regression response:', resultLogisticRegression);

          setClassificationResult({
              naiveBayes: resultNaiveBayes.classification,
              logisticRegression: resultLogisticRegression.classification,
          });

          setComparisonResult({
              naiveBayes: {
                  accuracy: resultNaiveBayes.accuracy,
                  precision: resultNaiveBayes.precision,
                  recall: resultNaiveBayes.recall,
                  f1_score: resultNaiveBayes.f1_score,
              },
              logisticRegression: {
                  accuracy: resultLogisticRegression.accuracy,
                  precision: resultLogisticRegression.precision,
                  recall: resultLogisticRegression.recall,
                  f1_score: resultLogisticRegression.f1_score,
              },
          });
      } catch (error) {
          console.error('Error classifying email:', error);
          setError(error.message);
      }
  };

  return (
    <ChakraProvider>
      <Box p={4}>
        <Heading mb={4}>Email Classifier</Heading>
        <Textarea
          value={emailText}
          onChange={(e) => setEmailText(e.target.value)}
          placeholder="Enter email text here..."
          size="sm"
          mb={4}
        />
        <Button colorScheme="blue" onClick={handleClassify}>Classify Email</Button>
        {error && (
          <Alert status="error" mt={4}>
            <AlertIcon />
            {error}
          </Alert>
        )}
        {classificationResult && (
          <VStack mt={4} spacing={4}>
            <Alert status="info">
              <AlertIcon />
              Naive Bayes Classification: {classificationResult.naiveBayes}
            </Alert>
            <Alert status="info">
              <AlertIcon />
              Logistic Regression Classification: {classificationResult.logisticRegression}
            </Alert>
          </VStack>
        )}
        {comparisonResult && (
          <Box mt={4}>
            <Heading size="md" mb={2}>Comparison Results</Heading>
            <HStack spacing={8}>
              <VStack>
                <Text fontWeight="bold">Naive Bayes</Text>
                <Text>Accuracy: {comparisonResult.naiveBayes.accuracy || 'N/A'}</Text>
                <Text>Precision: {comparisonResult.naiveBayes.precision || 'N/A'}</Text>
                <Text>Recall: {comparisonResult.naiveBayes.recall || 'N/A'}</Text>
                <Text>F1 Score: {comparisonResult.naiveBayes.f1_score || 'N/A'}</Text>
              </VStack>
              <VStack>
                <Text fontWeight="bold">Logistic Regression</Text>
                <Text>Accuracy: {comparisonResult.logisticRegression.accuracy || 'N/A'}</Text>
                <Text>Precision: {comparisonResult.logisticRegression.precision || 'N/A'}</Text>
                <Text>Recall: {comparisonResult.logisticRegression.recall || 'N/A'}</Text>
                <Text>F1 Score: {comparisonResult.logisticRegression.f1_score || 'N/A'}</Text>
              </VStack>
            </HStack>
          </Box>
        )}
      </Box>
    </ChakraProvider>
  );
}

export default App;
