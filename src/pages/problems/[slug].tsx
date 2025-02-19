import { useRouter } from "next/router";
import { api } from "~/utils/api";
import {
  Box,
  Heading,
  Text,
  HStack,
  Spinner,
  Code,
  VStack,
  Button,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Tabs,
  TabList,
  TabPanels,
  TabPanel,
  Tab,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
} from "@chakra-ui/react";
import { useState, useEffect } from "react";
import { Layout } from "~/components/layout";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import Editor from "@monaco-editor/react";

export default function ProblemPage() {
  const router = useRouter();
  const { slug } = router.query;
  const toast = useToast();
  const [code, setCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] = useState<{
    status: string | null;
    runtime: number | null;
    memory: number | null;
    passedTests: number | null;
    totalTests: number | null;
    stage: "CHECKING" | "BENCHMARKING" | "COMPLETED" | null;
    message: string | null;
  } | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);

  const submissionStatusQuery = api.problems.getSubmissionStatus.useQuery(
    { submissionId: submissionId! },
    { 
      enabled: !!submissionId,
      refetchInterval: 1000,
      refetchIntervalInBackground: true,
    }
  );

  const submissionsQuery = api.problems.getSubmissions.useQuery(
    { problemSlug: slug as string, limit: 10 },
    { enabled: !!slug }
  );

  const handleSubmit = () => {
    setIsSubmitting(true);
    setSubmissionStatus({
      status: "PENDING",
      runtime: null,
      memory: null,
      passedTests: null,
      totalTests: null,
      stage: "CHECKING",
      message: "Running test cases...",
    });
    submitMutation.mutate({
      problemSlug: slug as string,
      code,
      language: "cuda",
    });
  };

  const { data: problem, isLoading } = api.problems.getById.useQuery(
    { slug: slug as string },
    { enabled: !!slug }
  );

  useEffect(() => {
    if (problem?.starterCode) {
      setCode(problem.starterCode);
    }
  }, [problem]);

  useEffect(() => {
    const data = submissionStatusQuery.data;
    if (data) {
      setSubmissionStatus({
        status: data.status,
        runtime: data.runtime,
        memory: data.memory,
        passedTests: data.passedTests,
        totalTests: data.totalTests,
        stage: data.status === "CHECKING" 
          ? "CHECKING"
          : data.status === "BENCHMARKING" 
          ? "BENCHMARKING" 
          : "COMPLETED",
        message: data.status === "CHECKING"
          ? "Running test cases..."
          : data.status === "BENCHMARKING"
          ? "Running performance benchmark..."
          : null
      });

      if (
        data.status === "COMPLETED" || 
        data.status === "ERROR" || 
        data.status === "WRONG_ANSWER"
      ) {
        setSubmissionId(null);
        setIsSubmitting(false);
        submissionsQuery.refetch();
      }
    }
  }, [submissionStatusQuery.data]);

  const submitMutation = api.problems.submit.useMutation({
    onSuccess: (data) => {
      setSubmissionId(data.id);
    },
    onError: (error) => {
      setIsSubmitting(false);
      setSubmissionStatus(null);
      toast({
        title: "Submission failed",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  if (isLoading) {
    return (
      <Layout title="Loading...">
        <Box display="flex" justifyContent="center" alignItems="center">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!problem) {
    return (
      <Layout title="Not Found">
        <Box p={8}>
          <Text>Problem not found</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={problem.title}>
      <HStack align="start" spacing={8} h="100%" maxH="calc(100vh - 120px)">
        {/* Problem Description */}
        <Box w="50%" h="100%" overflowY="auto" p={6}>
          <Heading size="lg" mb={2}>
            {problem.title}
          </Heading>
          <Text color="gray.400" mb={6}>
            Difficulty: {problem.difficulty}
          </Text>

          <Box className="markdown" color="gray.100">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeHighlight]}
              components={{
                h1: (props) => (
                  <Heading as="h1" size="xl" mt={6} mb={4} {...props} />
                ),
                h2: (props) => (
                  <Heading as="h2" size="lg" mt={5} mb={3} {...props} />
                ),
                h3: (props) => (
                  <Heading as="h3" size="md" mt={4} mb={2} {...props} />
                ),
                ul: (props) => <Box as="ul" pl={8} mb={4} {...props} />,
                ol: (props) => <Box as="ol" pl={8} mb={4} {...props} />,
                li: (props) => <Box as="li" pl={2} mb={2} {...props} />,
                code: (props) => (
                  <Code
                    px={2}
                    py={1}
                    bg="gray.800"
                    color="gray.100"
                    borderRadius="md"
                    {...props}
                  />
                ),
                pre: (props) => (
                  <Box
                    as="pre"
                    p={4}
                    bg="gray.800"
                    borderRadius="md"
                    overflowX="auto"
                    mb={4}
                    {...props}
                  />
                ),
              }}
            >
              {problem.description}
            </ReactMarkdown>
          </Box>

          <Heading size="md" mb={4}>
            Example Test Cases
          </Heading>
          <VStack spacing={4} align="stretch">
            {problem.testCases.map((testCase) => (
              <Box key={testCase.id} p={4} bg="gray.800" borderRadius="md">
                <Text color="gray.300" mb={2}>
                  <Text as="span" fontWeight="bold" color="gray.200">
                    Input:{" "}
                  </Text>
                  <Code bg="gray.700" px={2}>
                    {testCase.input}
                  </Code>
                </Text>
                <Text color="gray.300">
                  <Text as="span" fontWeight="bold" color="gray.200">
                    Expected:{" "}
                  </Text>
                  <Code bg="gray.700" px={2}>
                    {testCase.expected}
                  </Code>
                </Text>
              </Box>
            ))}
          </VStack>
        </Box>

        {/* Code Editor and Submission */}
        <VStack w="50%" h="100%" spacing={4}>
          <Box w="100%" h="calc(100% - 300px)" bg="gray.800" borderRadius="xl" overflow="hidden">
            <Editor
              height="100%"
              defaultLanguage="cpp"
              theme="vs-dark"
              value={code}
              onChange={(value) => setCode(value ?? "")}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: "on",
                scrollBeyondLastLine: false,
                automaticLayout: true,
                padding: { top: 16, bottom: 16 },
                fontFamily: "JetBrains Mono, monospace",
              }}
            />
          </Box>

          <Box w="100%" bg="gray.800" borderRadius="xl" overflow="hidden">
            <Tabs variant="enclosed" colorScheme="blue">
              <TabList bg="gray.900" px={4} pt={2}>
                <Tab _selected={{ bg: "gray.800", borderBottomColor: "transparent" }}>Submit</Tab>
                <Tab _selected={{ bg: "gray.800", borderBottomColor: "transparent" }}>Submissions</Tab>
              </TabList>

              <TabPanels>
                <TabPanel p={4}>
                  <Button
                    colorScheme="blue"
                    size="lg"
                    width="100%"
                    onClick={handleSubmit}
                    isLoading={isSubmitting}
                    loadingText="Submitting..."
                    mb={4}
                  >
                    Submit Solution
                  </Button>

                  {submissionStatus && (
                    <Alert
                      status={
                        submissionStatus.status === "ACCEPTED"
                          ? "success"
                          : submissionStatus.status === "PENDING"
                          ? "info"
                          : "error"
                      }
                      variant="subtle"
                      flexDirection="column"
                      alignItems="flex-start"
                      borderRadius="md"
                      p={4}
                    >
                      <AlertIcon />
                      <AlertTitle mb={2}>
                        Status: {submissionStatus.status}
                      </AlertTitle>
                      <AlertDescription>
                        <VStack align="start" spacing={2}>
                          <Box p={3} bg="blackAlpha.300" borderRadius="md" w="100%">
                            {submissionStatus.stage && (
                              <Text fontWeight="bold" mb={2}>
                                Current Stage: {submissionStatus.stage}
                                {submissionStatus.message && (
                                  <Text color="gray.400" fontSize="sm" mt={1}>
                                    {submissionStatus.message}
                                  </Text>
                                )}
                              </Text>
                            )}
                            {submissionStatus.passedTests !== null && (
                              <Text fontWeight="bold" mb={2}>
                                Passed Tests: {submissionStatus.passedTests} / {submissionStatus.totalTests ?? "N/A"}
                              </Text>
                            )}
                            {submissionStatus.runtime && (
                              <Text>Execution Time: {submissionStatus.runtime.toFixed(2)}ms</Text>
                            )}
                            {submissionStatus.memory && (
                              <Text>Memory Usage: {submissionStatus.memory.toFixed(2)}MB</Text>
                            )}
                          </Box>
                        </VStack>
                      </AlertDescription>
                    </Alert>
                  )}
                </TabPanel>

                <TabPanel p={4}>
                  {submissionsQuery.isLoading ? (
                    <Box display="flex" justifyContent="center" p={4}>
                      <Spinner />
                    </Box>
                  ) : submissionsQuery.data?.submissions.length === 0 ? (
                    <Text color="gray.400" textAlign="center">
                      No submissions yet
                    </Text>
                  ) : (
                    <Box overflowX="auto">
                      <Table variant="simple" size="sm">
                        <Thead>
                          <Tr>
                            <Th>Status</Th>
                            <Th>Runtime</Th>
                            <Th>Memory</Th>
                            <Th>Passed Tests</Th>
                            <Th>Submitted</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {submissionsQuery.data?.submissions.map((submission) => (
                            <Tr key={submission.id}>
                              <Td>
                                <Badge
                                  colorScheme={
                                    submission.status === "ACCEPTED"
                                      ? "green"
                                      : submission.status === "PENDING"
                                      ? "yellow"
                                      : "red"
                                  }
                                >
                                  {submission.status}
                                </Badge>
                              </Td>
                              <Td>{submission.runtime ? `${submission.runtime}ms` : "-"}</Td>
                              <Td>{submission.memory ? `${submission.memory}MB` : "-"}</Td>
                              <Td>{submission.passedTests ? `${submission.passedTests} / ${submission.totalTests}` : "-"}</Td>
                              <Td>
                                {new Date(submission.createdAt).toLocaleString()}
                              </Td>
                            </Tr>
                          ))}
                        </Tbody>
                      </Table>
                    </Box>
                  )}
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Box>
        </VStack>
      </HStack>
    </Layout>
  );
}
