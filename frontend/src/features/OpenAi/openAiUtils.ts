import OpenAI from "openai";

const OPEN_AI_API_KEY = process.env.NEXT_PUBLIC_OPEN_AI_API_KEY_DLP002;

export async function askChatGpt(
  question: string
): Promise<string | undefined> {
  if (!OPEN_AI_API_KEY) throw new Error("OPEN_AI_API_KEY is missing");

  const openai = new OpenAI({
    apiKey: OPEN_AI_API_KEY,
  });
  const completion = await openai.completions.create({
    model: "text-davinci-003",
    prompt: question,
    temperature: 0.6,
    max_tokens: 1000,
  });
  return completion.choices[0].text;
}
