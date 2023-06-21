import { Configuration, OpenAIApi } from "openai";

const OPEN_AI_API_KEY = process.env.NEXT_PUBLIC_OPEN_AI_API_KEY_DLP002;

export async function askChatGpt(
  question: string
): Promise<string | undefined> {
  const configuration = new Configuration({
    apiKey: OPEN_AI_API_KEY,
  });
  const openai = new OpenAIApi(configuration);
  const completion = await openai.createCompletion({
    model: "text-davinci-003",
    prompt: question,
    temperature: 0.6,
    max_tokens: 1000,
  });
  return completion.data.choices[0].text;
}
