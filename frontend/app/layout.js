import './layout.css';
import './globals.css';

export const metadata = {
  title: "Medical Report Intelligence",
  description: "Advanced clinical analysis and document intelligence dashboard.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}