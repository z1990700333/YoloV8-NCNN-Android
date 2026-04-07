@rem Gradle wrapper script for Windows
@if "%DEBUG%"=="" @echo off
set DIRNAME=%~dp0
set CLASSPATH=%DIRNAME%\gradle\wrapper\gradle-wrapper.jar
set JAVACMD=java
if defined JAVA_HOME set JAVACMD=%JAVA_HOME%\bin\java
"%JAVACMD%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
