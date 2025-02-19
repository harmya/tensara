import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { id } = req.query;

  if (req.method !== "GET") {
    res.setHeader("Allow", ["GET"]);
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    // Initial check
    const initialSubmission = await db.submission.findUnique({
      where: { id: id as string },
    });

    if (!initialSubmission) {
      res.status(404).end();
      return;
    }

    // Send initial state
    res.write(`data: ${JSON.stringify(initialSubmission)}\n\n`);

    // Create interval to check submission status
    const interval = setInterval(async () => {
      try {
        const submission = await db.submission.findUnique({
          where: { id: id as string },
        });

        if (!submission) {
          clearInterval(interval);
          res.end();
          return;
        }

        res.write(`data: ${JSON.stringify(submission)}\n\n`);

        if (
          submission.status &&
          ["ACCEPTED", "ERROR", "WRONG_ANSWER", "TIMEOUT"].includes(
            submission.status
          )
        ) {
          clearInterval(interval);
          res.end();
        }
      } catch (error) {
        console.error("Error in SSE interval:", error);
        clearInterval(interval);
        res.end();
      }
    }, 1000);

    // Clean up on client disconnect
    req.on("close", () => {
      clearInterval(interval);
    });
  } catch (error) {
    console.error("Error in SSE handler:", error);
    res.status(500).end();
  }
}
