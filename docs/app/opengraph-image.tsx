import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const contentType = 'image/png';
export const size = {
  width: 1200,
  height: 630,
};

export default async function Image() {
  try {
    // Load the blue logo
    const logoBlueData = await fetch(new URL('/logo-blue.png', process.env.NEXT_PUBLIC_BASE_URL || 'https://struktx.vercel.app')).then(
      (res) => res.arrayBuffer(),
    );

    return new ImageResponse(
      (
        <div
          style={{
            height: '100%',
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#ffffff',
          }}
        >
          <img
            src={logoBlueData as any}
            alt="StruktX"
            width={400}
            height={400}
          />
        </div>
      ),
      {
        ...size,
      },
    );
  } catch (e: any) {
    console.log(`${e.message}`);
    return new Response(`Failed to generate the image`, {
      status: 500,
    });
  }
}
