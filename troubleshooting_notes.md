# Initial installation through downloaded Aware-Light

1. Enable all requests for access otherwise the application will flicker - these accesses can be modified later on when users managed to access the application.
2. Sign up using any "user_id" of users' choice, it has no additional meaning and will be encrypted during storage.
3. Turn off power saving mode.

# Build and Compile Locally using Android Studio

1. Run `git submodule update --init` at the terminal to load the plugin submodules.
2. Clean and rebuild the project (or run `./gradlew.bat clean build` at the terminal).
3. Check compatibility of gradle and Java version if error occurs.
4. Start the emulator and run `./gradlew.bat installDebug` to install the application in emulator/device.
5. Generate a signed APK for distribution: Run `./gradlew.bat assembleRelease` at the terminal.
