import java.io.{File, FileWriter}

val fileWriter = new FileWriter(new File("/tmp/hello.txt"))
fileWriter.write("hello there")
fileWriter.close()