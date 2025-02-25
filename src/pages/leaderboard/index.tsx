import { type NextPage } from "next";
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Badge,
  Link as ChakraLink,
  Tooltip,
  Spinner,
  Flex,
  Heading,
  Select,
  HStack,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
} from "@chakra-ui/react";
import { useState } from "react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { useRouter } from "next/router";

const GPU_DISPLAY_NAMES: Record<string, string> = {
  "T4": "NVIDIA T4",
  "H100": "NVIDIA H100",
  "A100": "NVIDIA A100",
  "A10G": "NVIDIA A10G",
  "L4": "NVIDIA L4",
  "all": "All GPUs"
};

const LeaderboardIndexPage: NextPage = () => {
  const router = useRouter();
  const [selectedGpu, setSelectedGpu] = useState<string>("all");

  const { data: problems, isLoading: isProblemsLoading } = api.problems.getAll.useQuery();
  const { data: submissions, isLoading: isSubmissionsLoading } = api.submissions.getAllSubmissions.useQuery();

  // Process submissions to get the best submission per user per problem per GPU type
  const getBestSubmissions = (problemSlug: string) => {
    if (!submissions) return [];

    const problemSubmissions = submissions.filter(
      (submission) => submission.problem.slug === problemSlug
    );

    const userGpuBestMap = new Map<string, typeof submissions[0]>();

    problemSubmissions.forEach((submission) => {
      if (submission.status !== "ACCEPTED" || !submission.gflops) return;
      
      if (selectedGpu !== "all" && submission.gpuType !== selectedGpu) return;

      const userGpuKey = `${submission.user.name ?? "Anonymous"}-${submission.gpuType}`;
      const currentBest = userGpuBestMap.get(userGpuKey);

      if (!currentBest || submission.gflops > currentBest.gflops!) {
        userGpuBestMap.set(userGpuKey, submission);
      }
    });

    return Array.from(userGpuBestMap.values()).sort(
      (a, b) => (b.gflops ?? 0) - (a.gflops ?? 0)
    );
  };

  if (isProblemsLoading || isSubmissionsLoading) {
    return (
      <Layout title="Leaderboard">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          h="50vh"
        >
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={`Leaderboards: ${GPU_DISPLAY_NAMES[selectedGpu]}`}>
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <HStack justify="space-between" align="center">
            <Heading size="lg">Leaderboards</Heading>
            <Select
              value={selectedGpu}
              onChange={(e) => setSelectedGpu(e.target.value)}
              w="200px"
              bg="whiteAlpha.50"
              color="white"
              borderColor="whiteAlpha.200"
              _hover={{ borderColor: "whiteAlpha.400" }}
              _focus={{ borderColor: "blue.500" }}
            >
              <option value="all">All GPUs</option>
              <option value="T4">NVIDIA T4</option>
              <option value="H100">NVIDIA H100</option>
              <option value="A10G">NVIDIA A10G</option>
              <option value="A100">NVIDIA A100</option>
              <option value="L4">NVIDIA L4</option>
            </Select>
          </HStack>

          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
            {problems?.map((problem) => {
              const topSubmissions = getBestSubmissions(problem.slug).slice(0, 3);
              
              return (
                <Card
                  key={problem.slug}
                  bg="gray.800"
                  borderColor="whiteAlpha.200"
                  borderWidth={1}
                  _hover={{ borderColor: "whiteAlpha.400" }}
                >
                  <CardHeader>
                    <ChakraLink
                      as={Link}
                      href={`/leaderboard/${problem.slug}${selectedGpu !== "all" ? `?gpu=${selectedGpu}` : ""}`}
                    >
                      <Heading size="md" color="white" _hover={{ color: "blue.400" }}>
                        {problem.title}
                      </Heading>
                    </ChakraLink>
                  </CardHeader>
                  <CardBody>
                    {topSubmissions.length === 0 ? (
                      <Text color="whiteAlpha.700">
                        No submissions yet{selectedGpu !== "all" ? ` for ${GPU_DISPLAY_NAMES[selectedGpu]}` : ""}
                      </Text>
                    ) : (
                      <Table variant="unstyled" size="sm">
                        <Thead>
                          <Tr>
                            <Th pl={2} color="whiteAlpha.600">Rank</Th>
                            <Th color="whiteAlpha.600">User</Th>
                            <Th color="whiteAlpha.600">GPU</Th>
                            <Th isNumeric color="whiteAlpha.600">GFLOPS</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {topSubmissions.map((submission, index) => (
                            <Tr key={submission.id} onClick={() => router.push(`/submissions/${submission.id}`)} cursor="pointer" 
                            _hover={{ bg: "gray.600", transform: "scale(1.02)", transition: "background-color 0.2s, transform 0.2s" }}
                            px={4}
                            rounded="full">
                              <Td pl={2}>
                                <Badge
                                  colorScheme={index === 0 ? "yellow" : index === 1 ? "gray" : "blue"}
                                  variant="solid"
                                  fontSize="sm"
                                  px={2}
                                >
                                  #{index + 1}
                                </Badge>
                              </Td>
                              <Td color="white">
                                {submission.user.name ?? "Anonymous"}
                              </Td>
                              <Td>
                                <Badge
                                  colorScheme="purple"
                                  variant="subtle"
                                  fontSize="xs"
                                >
                                  {submission.gpuType}
                                </Badge>
                              </Td>
                              <Td isNumeric>
                                <Tooltip label={`Runtime: ${submission.runtime?.toFixed(2)} ms`}>
                                  <Text color="green.300" fontWeight="semibold">
                                    {submission.gflops?.toFixed(2)}
                                  </Text>
                                </Tooltip>
                              </Td>
                            </Tr>
                          ))}
                        </Tbody>
                      </Table>
                    )}
                  </CardBody>
                </Card>
              );
            })}
          </SimpleGrid>
        </Flex>
      </Box>
    </Layout>
  );
};

export default LeaderboardIndexPage;
