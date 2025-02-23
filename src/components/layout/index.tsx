import { ReactNode, useState } from "react";
import Head from "next/head";
import { Box, Grid } from "@chakra-ui/react";
import { Header } from "./header";
import { Sidebar } from "./sidebar";

interface LayoutProps {
  title?: string;
  children: ReactNode;
  ogDescription?: string;
}

export function Layout({
  title = "Tensara",
  children,
  ogDescription = "Competitive programming platform",
}: LayoutProps) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content={ogDescription} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <Box minH="100vh" bg="gray.900" p={4}>
        <Grid
          templateAreas={`
            "header header"
            "sidebar main"
          `}
          gridTemplateRows={"72px 1fr"}
          gridTemplateColumns={`${isSidebarCollapsed ? "80px" : "280px"} 1fr`}
          gap={4}
          h="calc(100vh - 32px)"
          transition="all 0.2s"
        >
          <Box gridArea="header">
            <Header />
          </Box>

          <Box gridArea="sidebar" transition="all 0.2s">
            <Sidebar isCollapsed={isSidebarCollapsed} onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)} />
          </Box>

          <Box gridArea="main" bg="brand.secondary" borderRadius="2xl" p={6}>
            {children}
          </Box>
        </Grid>
      </Box>
    </>
  );
}
