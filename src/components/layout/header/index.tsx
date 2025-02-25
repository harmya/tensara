import { Box, Flex, HStack, Text, Image, Button, Icon, IconButton, useDisclosure, VStack, Drawer, DrawerBody, DrawerHeader, DrawerOverlay, DrawerContent, DrawerCloseButton } from "@chakra-ui/react";
import { useSession, signIn, signOut } from "next-auth/react";
import Link from "next/link";
import { useRouter } from "next/router";
import { FiCode, FiList, FiBookOpen, FiLogOut, FiGithub, FiMenu } from "react-icons/fi";
import { useState, useEffect } from "react";

export function Header() {
  const { data: session } = useSession();
  const router = useRouter();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const navItems = [
    { label: "Problems", href: "/problems", icon: FiCode },
    { label: "Submissions", href: "/submissions", icon: FiList },
    { label: "Blog", href: "/blog", icon: FiBookOpen },
  ];

  const isActivePath = (path: string) => {
    if (path === "/problems") {
      return (
        router.pathname === "/problems" ||
        router.pathname.startsWith("/problems/")
      );
    }
    return router.pathname === path;
  };

  const handleSignIn = () => {
    signIn("github", { callbackUrl: router.asPath }).catch(console.error);
  };

  const handleSignOut = () => {
    signOut({ callbackUrl: "/" }).catch(console.error);
  };

  const NavLinks = () => (
    <>
      {navItems.map((item) => (
        <Link key={item.href} href={item.href} passHref legacyBehavior>
          <Button
            as="a"
            variant="ghost"
            color="white"
            px={3}
            bg={isActivePath(item.href) ? "whiteAlpha.200" : "transparent"}
            _hover={{
              textDecoration: "none",
              bg: "whiteAlpha.100",
            }}
            fontSize="sm"
            leftIcon={<Icon as={item.icon} boxSize={4} />}
            w={isMobile ? "full" : "auto"}
          >
            {item.label}
          </Button>
        </Link>
      ))}
    </>
  );

  const AuthSection = () => (
    <>
      {session ? (
        <HStack spacing={4}>
          <HStack spacing={3}>
            <Image
              src={session.user?.image ?? ""}
              alt="Profile"
              w={8}
              h={8}
              rounded="full"
            />
            <Text color="white" fontSize="sm">
              {session.user?.name}
            </Text>
          </HStack>
          <Button
            variant="ghost"
            size="sm"
            color="white"
            onClick={handleSignOut}
            leftIcon={<Icon as={FiLogOut} boxSize={4} />}
            _hover={{
              bg: "whiteAlpha.200",
            }}
          >
            Sign out
          </Button>
        </HStack>
      ) : (
        <Button
          variant="ghost"
          color="white"
          onClick={handleSignIn}
          leftIcon={<Icon as={FiGithub} boxSize={5} />}
          bg="#24292e"
          _hover={{
            bg: "#2f363d",
          }}
        >
          Sign in with GitHub
        </Button>
      )}
    </>
  );

  return (
    <Box bg="brand.navbar" h="full" borderRadius="xl" px={6} py={2}>
      <Flex h="full" alignItems="center" justifyContent="space-between">
        <HStack spacing={8}>
          <Link href="/" passHref legacyBehavior>
            <Text
              as="a"
              fontSize="lg"
              fontWeight="bold"
              color="white"
              _hover={{ textDecoration: "none" }}
            >
              tensara
            </Text>
          </Link>

          {/* Desktop Navigation */}
          {!isMobile && (
            <HStack spacing={2}>
              <NavLinks />
            </HStack>
          )}
        </HStack>

        {/* Mobile Menu Button */}
        {isMobile && (
          <IconButton
            aria-label="Open menu"
            icon={<FiMenu />}
            variant="ghost"
            color="white"
            onClick={onOpen}
          />
        )}

        {/* Desktop Auth Section */}
        {!isMobile && <AuthSection />}

        {/* Mobile Drawer */}
        <Drawer isOpen={isOpen} placement="right" onClose={onClose}>
          <DrawerOverlay />
          <DrawerContent bg="gray.900">
            <DrawerCloseButton color="white" />
            <DrawerHeader borderBottomWidth="1px" color="white">
              Menu
            </DrawerHeader>
            <DrawerBody>
              <VStack align="stretch" spacing={4} mt={4}>
                <NavLinks />
                <Box pt={4} borderTopWidth="1px">
                  <AuthSection />
                </Box>
              </VStack>
            </DrawerBody>
          </DrawerContent>
        </Drawer>
      </Flex>
    </Box>
  );
}

// function NavLink({
//   href,
//   children,
// }: {
//   href: string;
//   children: React.ReactNode;
// }) {
//   return (
//     <Link href={href} passHref>
//       <Text
//         px={2}
//         py={1}
//         rounded="md"
//         color="gray.300"
//         _hover={{
//           textDecoration: "none",
//           bg: "gray.700",
//           color: "white",
//         }}
//         fontSize="sm"
//         fontWeight="medium"
//       >
//         {children}
//       </Text>
//     </Link>
//   );
// }
